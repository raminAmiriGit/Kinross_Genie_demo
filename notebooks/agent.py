
import uuid
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from mlflow.models import set_model
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
PROD_SPACE = "01f1bc7cf6ad18c8808fb150a484b5bb"
ASSET_SPACE = "01f1bc7cf6dd1ca480ab53935466040f"
VS_INDEX = "ramin_serverless_aws_catalog.kinross_mining.maintenance_reports_index"
RECOVERY_EP = "kinross-recovery-prediction"
ANOMALY_EP = "kinross-vibration-anomaly"

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
