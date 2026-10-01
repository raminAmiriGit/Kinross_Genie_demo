# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Genie-1 supervisor agent
# MAGIC Builds the multi-agent supervisor with five tools — the two Genie spaces, the two
# MAGIC ML model endpoints, and the inspection-report retriever — logs it as an MLflow
# MAGIC ChatAgent with the resource grants it needs, registers it to Unity Catalog, and
# MAGIC deploys it to a serving endpoint via `databricks.agents`.

# COMMAND ----------
# MAGIC %pip install -U databricks-langchain langgraph databricks-agents mlflow databricks-vectorsearch langchain-core -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
dbutils.widgets.text("llm_endpoint", "databricks-claude-sonnet-4-5")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
FQ = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{FQ}.genie_one"
VS_INDEX = f"{FQ}.maintenance_reports_index"
RECOVERY_EP = "kinross-recovery-prediction"
ANOMALY_EP = "kinross-vibration-anomaly"

# COMMAND ----------
# MAGIC %md ## Resolve the Genie space IDs from the registry
# COMMAND ----------
reg = {r["key"]: r["value"] for r in spark.table(f"{FQ}.demo_registry").collect()}
PROD_SPACE = reg["production_genie_space_id"]
ASSET_SPACE = reg["asset_genie_space_id"]
print("production space:", PROD_SPACE, "| asset space:", ASSET_SPACE)

# COMMAND ----------
# MAGIC %md ## Write the agent (models-from-code)
# COMMAND ----------
agent_code = '''
import uuid
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from mlflow.models import set_model
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

LLM_ENDPOINT = "__LLM__"
PROD_SPACE = "__PROD__"
ASSET_SPACE = "__ASSET__"
VS_INDEX = "__VS_INDEX__"
RECOVERY_EP = "__RECOVERY_EP__"
ANOMALY_EP = "__ANOMALY_EP__"

llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
_prod = GenieAgent(genie_space_id=PROD_SPACE, genie_agent_name="ProductionGenie",
                   description="Gold production and metallurgy analytics by site and date.")
_asset = GenieAgent(genie_space_id=ASSET_SPACE, genie_agent_name="AssetGenie",
                    description="Equipment health, maintenance and inspection analytics.")

@tool
def ask_production(question: str) -> str:
    "Ask the Production and Metallurgy Genie about gold recovery, head grade, grind size P80 and throughput by site and date."
    r = _prod.invoke({"messages": [{"role": "user", "content": question}]})
    return r["messages"][-1].content

@tool
def ask_asset(question: str) -> str:
    "Ask the Asset and Maintenance Genie about equipment health, vibration, bearing temperature, work orders, condition monitoring and inspection reports."
    r = _asset.invoke({"messages": [{"role": "user", "content": question}]})
    return r["messages"][-1].content

@tool
def predict_recovery(head_grade_gpt: float, grind_p80_um: float, throughput_tph: float) -> str:
    "Predict expected gold recovery percent for a feed grade (g/t), grind size P80 (microns) and throughput (t/h). Use to quantify the recovery gap between spec grind (about 106 um) and the current grind."
    w = WorkspaceClient()
    r = w.serving_endpoints.query(RECOVERY_EP, dataframe_records=[
        {"head_grade_gpt": head_grade_gpt, "grind_p80_um": grind_p80_um, "throughput_tph": throughput_tph}])
    return str(r.predictions)

@tool
def score_anomaly(vibration_mm_s: float, bearing_temp_c: float, power_kw: float) -> str:
    "Score whether an equipment sensor reading is anomalous, given vibration (mm/s), bearing temperature (C) and power draw (kW). The vibration alarm threshold is 6.0 mm/s."
    w = WorkspaceClient()
    r = w.serving_endpoints.query(ANOMALY_EP, dataframe_records=[
        {"vibration_mm_s": vibration_mm_s, "bearing_temp_c": bearing_temp_c, "power_kw": power_kw}])
    return str(r.predictions)

retriever = VectorSearchRetrieverTool(
    index_name=VS_INDEX, num_results=3, tool_name="search_inspection_reports",
    tool_description="Search maintenance inspection reports for findings, severity and recommendations such as bearing vibration warnings.")

SYSTEM = ("You are Genie-1, the supervisor analytics agent for Kinross gold mining operations. "
    "You orchestrate specialist tools. Routing: use ask_production for gold recovery, head grade, "
    "grind size P80 and throughput questions by site or date; use ask_asset for equipment health, "
    "vibration, work orders, condition monitoring and inspection questions; use predict_recovery to "
    "quantify how much recovery is lost from a coarser grind; use score_anomaly to check if a sensor "
    "reading is anomalous; use search_inspection_reports to find written inspection findings and "
    "recommendations. When asked why recovery is down, first use ask_production to find the site and "
    "equipment driving it, then use ask_asset and search_inspection_reports to find the root cause, and "
    "use predict_recovery to quantify the impact. Sites: Paracatu (PARA), Tasiast (TASI), Fort Knox "
    "(FKNX). Always state the site, the equipment, the root cause and the quantified impact, and note "
    "which source each fact came from. Be concise.")

_graph = create_react_agent(llm, [ask_production, ask_asset, predict_recovery, score_anomaly, retriever], prompt=SYSTEM)

class Genie1(ChatAgent):
    def predict(self, messages, context=None, custom_inputs=None):
        req = {"messages": [{"role": m.role, "content": m.content} for m in messages]}
        result = _graph.invoke(req)
        final = result["messages"][-1]
        content = getattr(final, "content", str(final))
        return ChatAgentResponse(messages=[ChatAgentMessage(role="assistant", content=content, id=uuid.uuid4().hex)])

set_model(Genie1())
'''
agent_code = (agent_code
    .replace("__LLM__", LLM_ENDPOINT).replace("__PROD__", PROD_SPACE).replace("__ASSET__", ASSET_SPACE)
    .replace("__VS_INDEX__", VS_INDEX).replace("__RECOVERY_EP__", RECOVERY_EP).replace("__ANOMALY_EP__", ANOMALY_EP))
with open("agent.py", "w") as f:
    f.write(agent_code)
print("wrote agent.py")

# COMMAND ----------
# MAGIC %md ## Log, register and deploy
# COMMAND ----------
import mlflow
from mlflow.models.resources import (DatabricksServingEndpoint, DatabricksVectorSearchIndex,
                                      DatabricksSQLWarehouse, DatabricksTable)
try:
    from mlflow.models.resources import DatabricksGenieSpace
    genie_resources = [DatabricksGenieSpace(genie_space_id=PROD_SPACE), DatabricksGenieSpace(genie_space_id=ASSET_SPACE)]
except Exception:
    genie_resources = []

mlflow.set_registry_uri("databricks-uc")
# Declare the warehouse + every table the Genie spaces query, so the agent's system
# service principal is auto-granted access to run the spaces' SQL (not just invoke them).
agent_tables = ["production_daily", "sites", "equipment", "sensor_readings", "work_orders",
                "condition_monitoring", "maintenance_reports_parsed", "mv_recovery",
                "mv_throughput", "mv_downtime", "v_gold_lost"]
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=RECOVERY_EP),
    DatabricksServingEndpoint(endpoint_name=ANOMALY_EP),
    DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en"),
    DatabricksVectorSearchIndex(index_name=VS_INDEX),
    DatabricksSQLWarehouse(warehouse_id="a5246712a94a1cce"),
] + [DatabricksTable(table_name=f"{FQ}.{t}") for t in agent_tables] + genie_resources

example = {"messages": [{"role": "user", "content": "Why is gold recovery down this week?"}]}
with mlflow.start_run(run_name="genie_one"):
    info = mlflow.pyfunc.log_model(
        name="agent", python_model="agent.py", resources=resources, input_example=example,
        pip_requirements=["databricks-langchain", "langgraph", "databricks-vectorsearch",
                          "langchain-core", "mlflow", "databricks-sdk"],
        registered_model_name=MODEL_NAME)
print("registered", MODEL_NAME, "version", info.registered_model_version)

# COMMAND ----------
from databricks import agents
dep = agents.deploy(MODEL_NAME, info.registered_model_version, tags={"demo": "kinross-genie"})
endpoint_name = dep.endpoint_name if hasattr(dep, "endpoint_name") else str(dep)
print("agent endpoint:", endpoint_name)

spark.sql(f"DELETE FROM {FQ}.demo_registry WHERE key = 'genie_one_endpoint'")
spark.sql(f"INSERT INTO {FQ}.demo_registry VALUES ('genie_one_endpoint', '{endpoint_name}')")
print("registry updated: genie_one_endpoint =", endpoint_name)
