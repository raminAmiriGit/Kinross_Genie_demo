# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Production & Metallurgy Genie space
# MAGIC Creates the structured Genie space for recovery, grade, grind and throughput.
# MAGIC Idempotent: patches the space if it already exists (matched by title). Writes the
# MAGIC space_id to the `demo_registry` table for the supervisor agent (notebook 10).

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
dbutils.widgets.text("warehouse_id", "a5246712a94a1cce")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
FQ = f"{CATALOG}.{SCHEMA}"
PARENT = "/Workspace/Users/ramin.amiri@databricks.com/Kinross_Genie_demo"
TITLE = "Kinross Production & Metallurgy Genie"

# COMMAND ----------
import json, uuid
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

def _id():
    return uuid.uuid4().hex

def relation(identifier):
    return {"identifier": identifier}

tables = [
    f"{FQ}.production_daily", f"{FQ}.sites", f"{FQ}.equipment", f"{FQ}.sensor_readings",
    f"{FQ}.mv_recovery", f"{FQ}.mv_throughput", f"{FQ}.v_gold_lost",
]

instructions = (
    "This space covers gold production and metallurgy across three sites: Paracatu (PARA), "
    "Tasiast (TASI) and Fort Knox (FKNX). Gold recovery (recovery_pct) is the percentage of "
    "contained gold recovered during processing; each site has a recovery_target_pct (Paracatu's "
    "target is 78%). Recovery falls when grind size (grind_p80_um, the 80th-percentile product "
    "particle size in microns) coarsens above target. Use v_gold_lost for ounces (gold_oz_lost) and "
    "USD (usd_lost) lost versus target. Prefer mv_recovery and mv_throughput for KPI rollups. When "
    "asked why recovery is down, compare recent recovery_pct and grind_p80_um by site and date and "
    "identify the site and equipment driving the change."
)
sample_questions = [
    "Why is gold recovery down this week?",
    "Which site has the lowest recovery and how many ounces were lost in the last 7 days?",
    "How has grind P80 at Paracatu changed over the last two weeks?",
]

serialized_space = {
    "version": 2,
    "data_sources": {"tables": sorted([relation(t) for t in tables], key=lambda r: r["identifier"])},
    "instructions": {"text_instructions": [{"id": _id(), "content": [instructions]}]},
    "config": {"sample_questions": [{"id": _id(), "question": [q]} for q in sample_questions]},
}

# COMMAND ----------
# MAGIC %md ## Create or patch (idempotent by title)
# COMMAND ----------
def list_spaces():
    out, token = [], None
    while True:
        params = {"page_size": 100}
        if token:
            params["page_token"] = token
        resp = w.api_client.do("GET", "/api/2.0/genie/spaces", query=params)
        out += resp.get("spaces", []) or []
        token = resp.get("next_page_token")
        if not token:
            break
    return out

existing = next((s for s in list_spaces() if s.get("title") == TITLE), None)
body = {
    "title": TITLE,
    "description": "Structured production & metallurgy analytics (recovery, grade, grind, throughput).",
    "parent_path": PARENT,
    "warehouse_id": WAREHOUSE_ID,
    "serialized_space": json.dumps(serialized_space),
}
if existing:
    space_id = existing["space_id"]
    w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)
    print("patched existing space:", space_id)
else:
    resp = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)
    space_id = resp["space_id"]
    print("created space:", space_id)

# COMMAND ----------
# MAGIC %md ## Register the space_id for downstream notebooks
# COMMAND ----------
spark.sql(f"CREATE TABLE IF NOT EXISTS {FQ}.demo_registry (key STRING, value STRING)")
spark.sql(f"DELETE FROM {FQ}.demo_registry WHERE key = 'production_genie_space_id'")
spark.sql(f"INSERT INTO {FQ}.demo_registry VALUES ('production_genie_space_id', '{space_id}')")
print("registry updated: production_genie_space_id =", space_id)
