# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Vector Search over the inspection reports
# MAGIC Builds a Vector Search index over `maintenance_reports_parsed.raw_text` so Genie-1
# MAGIC can **retrieve and cite the exact inspection finding** (the ignored SAG Mill #2
# MAGIC warning). This is the retrieval tool the supervisor agent calls.

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
VS_ENDPOINT = "kinross-vs"
INDEX = f"{FQ}.maintenance_reports_index"
SOURCE = f"{FQ}.maintenance_reports_parsed"
EMB_MODEL = "databricks-gte-large-en"

# COMMAND ----------
# MAGIC %md ## Enable Change Data Feed on the source (required for delta-sync index)
# COMMAND ----------
spark.sql(f"ALTER TABLE {SOURCE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("CDF enabled on", SOURCE)

# COMMAND ----------
# MAGIC %md ## Create the Vector Search endpoint (wait until ONLINE)
# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient
import time
vsc = VectorSearchClient(disable_notice=True)

def endpoint_exists(name):
    try:
        return any(e.get("name") == name for e in vsc.list_endpoints().get("endpoints", []))
    except Exception:
        return False

if not endpoint_exists(VS_ENDPOINT):
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print("creating endpoint", VS_ENDPOINT)

# poll until online (up to ~20 min)
for _ in range(60):
    try:
        st = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state", "")
        print("endpoint state:", st)
        if st == "ONLINE":
            break
    except Exception as e:
        print("waiting...", str(e)[:80])
    time.sleep(20)

# COMMAND ----------
# MAGIC %md ## Create (or sync) the delta-sync index with managed embeddings
# COMMAND ----------
def index_exists(name):
    try:
        return any(i.get("name") == name for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", []))
    except Exception:
        return False

if index_exists(INDEX):
    print("index exists, triggering sync")
    vsc.get_index(VS_ENDPOINT, INDEX).sync()
else:
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=INDEX,
        source_table_name=SOURCE,
        primary_key="report_id",
        pipeline_type="TRIGGERED",
        embedding_source_column="raw_text",
        embedding_model_endpoint_name=EMB_MODEL,
    )
    print("creating index", INDEX)

# COMMAND ----------
# MAGIC %md ## Wait for index ready, then test a retrieval
# COMMAND ----------
idx = vsc.get_index(VS_ENDPOINT, INDEX)
for _ in range(60):
    try:
        desc = idx.describe()
        ready = desc.get("status", {}).get("ready", False)
        detail = desc.get("status", {}).get("detailed_state", "")
        print("index ready:", ready, "|", detail)
        if ready:
            break
    except Exception as e:
        print("waiting for index...", str(e)[:80])
    time.sleep(20)

# COMMAND ----------
res = idx.similarity_search(
    query_text="high spindle bearing vibration on SAG Mill #2",
    columns=["report_id", "equipment_id", "severity", "recommendation"],
    num_results=3)
print(res)
