# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · End-to-end smoke test
# MAGIC Verifies every layer: tables, metric views, both model endpoints, the Vector Search
# MAGIC index, and the deployed Genie-1 agent answering the two demo questions. Fails loudly
# MAGIC if anything is missing.

# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
VS_INDEX = f"{FQ}.maintenance_reports_index"
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))

# COMMAND ----------
# MAGIC %md ## Tables & views
# COMMAND ----------
for t in ["sites", "equipment", "production_daily", "sensor_readings", "work_orders",
          "condition_monitoring", "maintenance_reports_parsed", "mining_glossary"]:
    try:
        n = spark.table(f"{FQ}.{t}").count()
        check(f"table {t}", n > 0, f"{n} rows")
    except Exception as e:
        check(f"table {t}", False, str(e)[:100])

try:
    dip = spark.sql(f"""SELECT recovery_pct FROM {FQ}.production_daily
                        WHERE site_id='PARA' ORDER BY production_date DESC LIMIT 1""").collect()[0][0]
    check("paracatu recovery dip < 74%", dip < 74, f"latest={round(dip,1)}%")
except Exception as e:
    check("paracatu recovery dip", False, str(e)[:100])

try:
    lost = spark.sql(f"""SELECT round(sum(usd_lost)) FROM {FQ}.v_gold_lost
                         WHERE site_id='PARA' AND production_date>=date_sub(current_date(),7)""").collect()[0][0]
    check("v_gold_lost impact computed", lost and lost > 0, f"7d USD lost={lost}")
except Exception as e:
    check("v_gold_lost", False, str(e)[:100])

# COMMAND ----------
# MAGIC %md ## Model endpoints
# COMMAND ----------
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
try:
    r = w.serving_endpoints.query("kinross-recovery-prediction", dataframe_records=[
        {"head_grade_gpt": 0.44, "grind_p80_um": 150.0, "throughput_tph": 4200.0}])
    check("recovery-prediction endpoint", r.predictions is not None, str(r.predictions)[:80])
except Exception as e:
    check("recovery-prediction endpoint", False, str(e)[:120])
try:
    r = w.serving_endpoints.query("kinross-vibration-anomaly", dataframe_records=[
        {"vibration_mm_s": 9.0, "bearing_temp_c": 76.0, "power_kw": 8200.0}])
    check("vibration-anomaly endpoint", r.predictions is not None, str(r.predictions)[:80])
except Exception as e:
    check("vibration-anomaly endpoint", False, str(e)[:120])

# COMMAND ----------
# MAGIC %md ## Vector Search retrieval
# COMMAND ----------
try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    idx = vsc.get_index("kinross-vs", VS_INDEX)
    hits = idx.similarity_search(query_text="high spindle bearing vibration on SAG Mill #2",
                                 columns=["report_id", "equipment_id", "severity"], num_results=3)
    rows = hits.get("result", {}).get("data_array", [])
    check("vector search retrieval", len(rows) > 0, f"{len(rows)} hits; top={rows[0] if rows else None}")
except Exception as e:
    check("vector search retrieval", False, str(e)[:120])

# COMMAND ----------
# MAGIC %md ## Genie-1 agent — the two demo questions
# COMMAND ----------
reg = {r["key"]: r["value"] for r in spark.table(f"{FQ}.demo_registry").collect()}
EP = reg.get("genie_one_endpoint")
print("agent endpoint:", EP)

agent_err = None
raw_resp = None
import requests
def ask(q):
    global raw_resp
    # Use a long timeout: the agent cold-starts and fans out to both Genie spaces.
    url = f"{w.config.host}/serving-endpoints/{EP}/invocations"
    r = requests.post(url, headers=w.config.authenticate(),
                      json={"messages": [{"role": "user", "content": q}]}, timeout=900)
    r.raise_for_status()
    resp = r.json()
    raw_resp = resp
    msgs = resp.get("messages", []) if isinstance(resp, dict) else []
    return " ".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")

if EP:
    def refused(ans):
        a = ans.lower()
        return any(p in a for p in ["unable to access", "permission restriction", "don't have access",
                                    "do not have access", "grant select", "database administrator"])
    try:
        a1 = ask("Why is gold recovery down this week?")
        print("\nQ1 answer:\n", a1[:1500])
        check("Genie-1 Q1 not a permission refusal", not refused(a1), "")
        check("Genie-1 Q1 identifies Paracatu", "paracatu" in a1.lower(), "")
        a2 = ask("What's going on with SAG Mill #2 at Paracatu?")
        print("\nQ2 answer:\n", a2[:1500])
        check("Genie-1 Q2 not a permission refusal", not refused(a2), "")
        check("Genie-1 Q2 mentions vibration/bearing", any(k in a2.lower() for k in ["vibration", "bearing"]), "")
    except Exception as e:
        agent_err = f"{type(e).__name__}: {e}"
        check("Genie-1 agent query", False, str(e)[:200])
else:
    check("Genie-1 endpoint registered", False, "genie_one_endpoint missing from registry")

# COMMAND ----------
# MAGIC %md ## Summary
# COMMAND ----------
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n==== {passed}/{len(results)} checks passed ====")
for name, ok, detail in results:
    print(("  PASS " if ok else "  FAIL ") + name + (f" | {detail}" if detail else ""))

import json as _json
summary = {
    "passed": passed, "total": len(results),
    "failed": [n for n, ok, _ in results if not ok],
    "q1": (a1[:600] if "a1" in dir() else None),
    "q2": (a2[:600] if "a2" in dir() else None),
    "agent_err": agent_err,
    "raw_resp": str(raw_resp)[:600],
}
dbutils.notebook.exit(_json.dumps(summary))
