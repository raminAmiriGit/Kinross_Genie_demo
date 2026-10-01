# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Asset & Maintenance Genie space
# MAGIC Creates the structured Genie space for equipment health, condition monitoring, work
# MAGIC orders and parsed inspection reports. Idempotent by title; registers its space_id.

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
dbutils.widgets.text("warehouse_id", "a5246712a94a1cce")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
FQ = f"{CATALOG}.{SCHEMA}"
PARENT = "/Workspace/Users/ramin.amiri@databricks.com/Kinross_Genie_demo"
TITLE = "Kinross Asset & Maintenance Genie"

# COMMAND ----------
import json, uuid
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
_id = lambda: uuid.uuid4().hex

tables = [
    f"{FQ}.equipment", f"{FQ}.work_orders", f"{FQ}.condition_monitoring",
    f"{FQ}.sensor_readings", f"{FQ}.maintenance_reports_parsed", f"{FQ}.mv_downtime",
]
instructions = (
    "This space covers equipment health and maintenance across Paracatu (PARA), Tasiast (TASI) and "
    "Fort Knox (FKNX). Equipment includes SAG mills, ball mills and crushers (see equipment). "
    "condition_monitoring records inspection readings; a reading above its threshold is an alarm "
    "(vibration alarm threshold is 6.0 mm/s). sensor_readings holds hourly time-series "
    "vibration_mm_s, bearing_temp_c and power_kw per equipment. work_orders tracks maintenance; a "
    "row with status 'Open' and priority 'High' is unresolved. maintenance_reports_parsed contains "
    "inspection report findings, severity and recommendations. When asked about a specific mill, "
    "check its recent vibration versus the 6.0 mm/s threshold, related condition_monitoring alerts, "
    "any open work orders, and inspection report findings."
)
sample_questions = [
    "What's going on with SAG Mill #2 at Paracatu?",
    "Show open high-priority work orders.",
    "Which equipment has vibration above the alarm threshold?",
]
serialized_space = {
    "version": 2,
    "data_sources": {"tables": sorted([{"identifier": t} for t in tables], key=lambda r: r["identifier"])},
    "instructions": {"text_instructions": [{"id": _id(), "content": [instructions]}]},
    "config": {"sample_questions": [{"id": _id(), "question": [q]} for q in sample_questions]},
}

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
    "description": "Equipment health, condition monitoring, work orders and inspection reports.",
    "parent_path": PARENT,
    "warehouse_id": WAREHOUSE_ID,
    "serialized_space": json.dumps(serialized_space),
}
if existing:
    space_id = existing["space_id"]
    w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)
    print("patched existing space:", space_id)
else:
    space_id = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)["space_id"]
    print("created space:", space_id)

# COMMAND ----------
spark.sql(f"CREATE TABLE IF NOT EXISTS {FQ}.demo_registry (key STRING, value STRING)")
spark.sql(f"DELETE FROM {FQ}.demo_registry WHERE key = 'asset_genie_space_id'")
spark.sql(f"INSERT INTO {FQ}.demo_registry VALUES ('asset_genie_space_id', '{space_id}')")
print("registry updated: asset_genie_space_id =", space_id)
