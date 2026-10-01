# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Metric Views (the ontology's governed KPIs)
# MAGIC Creates Unity Catalog **Metric Views** for recovery, throughput and downtime so
# MAGIC Genie answers use governed definitions, plus a plain `v_gold_lost` view that does
# MAGIC the financial-impact math. Falls back to standard views if metric-view DDL is
# MAGIC unavailable.

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------
# MAGIC %md ## Metric views (with graceful fallback to standard views)
# COMMAND ----------
metric_views = {
    "mv_recovery": dict(
        source=f"{FQ}.production_daily",
        yaml=f"""version: 0.1
source: {FQ}.production_daily
dimensions:
  - name: production_date
    expr: production_date
  - name: site_id
    expr: site_id
measures:
  - name: avg_recovery_pct
    expr: AVG(recovery_pct)
  - name: avg_grind_p80_um
    expr: AVG(grind_p80_um)
  - name: total_gold_oz
    expr: SUM(gold_oz)
""",
        fallback=f"""SELECT production_date, site_id,
                     AVG(recovery_pct) AS avg_recovery_pct,
                     AVG(grind_p80_um) AS avg_grind_p80_um,
                     SUM(gold_oz)      AS total_gold_oz
              FROM {FQ}.production_daily GROUP BY production_date, site_id"""),
    "mv_throughput": dict(
        source=f"{FQ}.production_daily",
        yaml=f"""version: 0.1
source: {FQ}.production_daily
dimensions:
  - name: production_date
    expr: production_date
  - name: site_id
    expr: site_id
measures:
  - name: avg_throughput_tph
    expr: AVG(throughput_tph)
  - name: total_ore_tonnes
    expr: SUM(ore_tonnes)
""",
        fallback=f"""SELECT production_date, site_id,
                     AVG(throughput_tph) AS avg_throughput_tph,
                     SUM(ore_tonnes)     AS total_ore_tonnes
              FROM {FQ}.production_daily GROUP BY production_date, site_id"""),
    "mv_downtime": dict(
        source=f"{FQ}.work_orders",
        yaml=f"""version: 0.1
source: {FQ}.work_orders
dimensions:
  - name: site_id
    expr: site_id
  - name: equipment_id
    expr: equipment_id
  - name: status
    expr: status
measures:
  - name: total_downtime_hours
    expr: SUM(downtime_hours)
  - name: work_order_count
    expr: COUNT(1)
""",
        fallback=f"""SELECT site_id, equipment_id, status,
                     SUM(downtime_hours) AS total_downtime_hours,
                     COUNT(1)            AS work_order_count
              FROM {FQ}.work_orders GROUP BY site_id, equipment_id, status"""),
}

for name, cfg in metric_views.items():
    fqn = f"{FQ}.{name}"
    try:
        spark.sql(f"CREATE OR REPLACE VIEW {fqn} WITH METRICS LANGUAGE YAML AS $${cfg['yaml']}$$")
        print(f"  metric view created: {fqn}")
    except Exception as e:
        print(f"  metric view DDL failed for {fqn} ({str(e)[:120]}); creating standard view")
        spark.sql(f"CREATE OR REPLACE VIEW {fqn} AS {cfg['fallback']}")

# COMMAND ----------
# MAGIC %md ## Financial impact view — ounces & USD lost vs target recovery
# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE VIEW {FQ}.v_gold_lost AS
SELECT
  p.production_date,
  p.site_id,
  s.site_name,
  round(p.recovery_pct, 2)                                   AS recovery_pct,
  s.recovery_target_pct,
  round(p.grind_p80_um, 1)                                   AS grind_p80_um,
  -- ounces that would have been recovered at target recovery, minus actual
  round(GREATEST(0.0, (s.recovery_target_pct - p.recovery_pct))/100.0
        * p.ore_tonnes * p.head_grade_gpt / 31.1035, 1)      AS gold_oz_lost,
  round(GREATEST(0.0, (s.recovery_target_pct - p.recovery_pct))/100.0
        * p.ore_tonnes * p.head_grade_gpt / 31.1035
        * p.gold_price_usd, 0)                               AS usd_lost
FROM {FQ}.production_daily p
JOIN {FQ}.sites s USING (site_id)
""")
spark.sql(f"COMMENT ON VIEW {FQ}.v_gold_lost IS "
          f"'Gold ounces and USD lost per day vs each site recovery target (impact math).'")
print("  created", f"{FQ}.v_gold_lost")

# COMMAND ----------
# MAGIC %md ## Verify — Paracatu impact over the last week
# COMMAND ----------
display(spark.sql(f"""
  SELECT site_name, sum(gold_oz_lost) oz_lost_7d, sum(usd_lost) usd_lost_7d
  FROM {FQ}.v_gold_lost
  WHERE production_date >= date_sub(current_date(), 7)
  GROUP BY site_name ORDER BY usd_lost_7d DESC
"""))
