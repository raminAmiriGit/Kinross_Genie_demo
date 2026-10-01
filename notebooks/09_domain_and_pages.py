# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · Mining domain + ontology (Pages)
# MAGIC Grounds the Genie **ontology** in governed business semantics:
# MAGIC 1. Rich Unity Catalog **comments** on the key tables/columns (Genie reads these).
# MAGIC 2. A `mining_glossary` table holding the **business-definition "Pages"** content.
# MAGIC 3. Best-effort creation of the **Mining domain** and tagging of assets (Public
# MAGIC    Preview, curator-gated — non-fatal if unavailable).
# MAGIC
# MAGIC Pages have no stable public create API yet, so this notebook prints the exact UI /
# MAGIC Genie Code bulk-import steps to promote the glossary into governed Pages and to
# MAGIC attach the Genie spaces + agent to the Mining domain.

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------
# MAGIC %md ## 1 · Business-semantic comments (ontology Genie reads directly)
# COMMAND ----------
comments = [
    (f"TABLE {FQ}.production_daily", "Daily gold production and metallurgy by site: ore, head grade, throughput, grind size, recovery and gold ounces."),
    (f"TABLE {FQ}.condition_monitoring", "Equipment condition-monitoring inspection readings; a reading above its threshold is an alarm."),
    (f"TABLE {FQ}.work_orders", "Maintenance work orders; status Open with priority High means an unresolved issue."),
    (f"TABLE {FQ}.sensor_readings", "Hourly equipment sensor time-series: vibration (mm/s), bearing temperature (C), power draw (kW)."),
    (f"TABLE {FQ}.maintenance_reports_parsed", "Parsed inspection reports with finding, severity and recommendation; raw_text holds the full document."),
]
col_comments = [
    (f"{FQ}.production_daily", "recovery_pct", "Gold recovery: percent of contained gold recovered in processing. Falls when grind coarsens above target."),
    (f"{FQ}.production_daily", "grind_p80_um", "Grind size P80: 80th-percentile product particle size in microns. Higher = coarser = worse liberation."),
    (f"{FQ}.production_daily", "gold_oz", "Gold ounces produced that day (ore x grade x recovery)."),
    (f"{FQ}.sensor_readings", "value", "Sensor reading value; the sensor_type column names the signal (vibration_mm_s alarms above 6.0 mm/s, bearing_temp_c, power_kw)."),
]
for target, c in comments:
    spark.sql(f"COMMENT ON {target} IS '{c.replace(chr(39), chr(39)+chr(39))}'")
for tbl, col, c in col_comments:
    spark.sql(f"ALTER TABLE {tbl} ALTER COLUMN {col} COMMENT '{c.replace(chr(39), chr(39)+chr(39))}'")
print("applied", len(comments), "table +", len(col_comments), "column comments")

# COMMAND ----------
# MAGIC %md ## 2 · mining_glossary — the Pages content (governed, queryable)
# COMMAND ----------
import pandas as pd
glossary = pd.DataFrame([
    ("Gold Recovery", "Percent of contained gold recovered during processing. Each site has a recovery target (Paracatu 78%). The headline metallurgical KPI.", "recovery; recovery rate; met recovery"),
    ("Grind Size P80", "80th-percentile particle size of mill product in microns. Coarser grind (higher P80) reduces gold liberation and lowers recovery.", "grind; P80; particle size; fineness"),
    ("SAG Mill", "Semi-autogenous grinding mill in the comminution circuit. Bearing or liner wear can coarsen the grind and hurt recovery.", "mill; grinding mill; SAG"),
    ("Throughput", "Ore processed per hour (t/h) through the mill circuit.", "tph; mill throughput; feed rate"),
    ("Condition Monitoring", "Scheduled equipment inspections and sensor readings. Vibration above 6.0 mm/s is an alarm indicating likely bearing wear.", "CM; vibration alarm; inspection"),
    ("Gold Lost / Financial Impact", "Ounces (and USD) not recovered versus target recovery: (target-actual)/100 x ore x grade / 31.1035, times gold price. See v_gold_lost.", "lost ounces; impact; usd lost"),
    ("Paracatu", "Large low-grade high-tonnage gold site (PARA), Brazil. Hero site of the recovery investigation.", "PARA"),
    ("Tasiast", "Gold site (TASI), Mauritania.", "TASI"),
    ("Fort Knox", "Gold site (FKNX), United States.", "FKNX"),
], columns=["term", "definition", "synonyms"])
(spark.createDataFrame(glossary)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.mining_glossary"))
spark.sql(f"COMMENT ON TABLE {FQ}.mining_glossary IS 'Business glossary (ontology Pages content) for the Mining domain.'")
print("wrote", f"{FQ}.mining_glossary", "rows:", glossary.shape[0])

# COMMAND ----------
# MAGIC %md ## 3 · Best-effort: create the Mining domain and tag assets (Preview)
# COMMAND ----------
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
domain_ok = False
try:
    if hasattr(w, "domains"):
        try:
            w.domains.create_domain(domain={"tag_key": "Mining",
                "description": "Kinross mining operations analytics assets.",
                "subtitle": "Recovery, assets and inspections"})
        except Exception:
            # some SDK builds take positional / different shape
            w.domains.create_domain({"tag_key": "Mining",
                "description": "Kinross mining operations analytics assets."})
        domain_ok = True
        print("Mining domain created (or already exists).")
    else:
        print("SDK build has no `domains` API; skipping programmatic domain creation.")
except Exception as e:
    print("Domain creation not available on this workspace (Preview / curator-gated):", str(e)[:160])

# COMMAND ----------
# MAGIC %md ## 4 · Manual steps to finish the ontology (printed for the demo owner)
# COMMAND ----------
msg = f"""
ONTOLOGY / DOMAIN — remaining UI steps (Pages have no public create API yet):

1. Catalog Explorer > Discover > Domains > create/open the 'Mining' domain.
   Attach assets: catalog {CATALOG} schema {SCHEMA} (tables production_daily, sensor_readings,
   condition_monitoring, work_orders, maintenance_reports_parsed, v_gold_lost, mv_recovery,
   mv_throughput, mv_downtime), both Genie spaces, and the deployed Genie-1 agent.

2. Create Pages from the glossary (or use Genie Code > Bulk import pages):
   source table = {FQ}.mining_glossary (term, definition, synonyms). Publish each as a Page.
   Certified Pages are cited as authoritative by Genie Ontology.

Programmatic grounding already in place: table/column comments + the metric views (03) +
the glossary table — Genie uses these today.
"""
print(msg)
