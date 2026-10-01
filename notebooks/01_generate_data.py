# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate synthetic mining data
# MAGIC Deterministic (seeded) synthetic data for 3 sites over 90 days. The **story is
# MAGIC planted here**: Paracatu recovery decays 78% → ~71% over the last ~10 days as the
# MAGIC **SAG Mill #2** grind coarsens, driven by a rising **vibration anomaly** that a
# MAGIC condition-monitoring inspection flagged and *nobody actioned*.
# MAGIC
# MAGIC Everything is synthetic — site names are for realism only.

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
TODAY = datetime.utcnow().date()
START = TODAY - timedelta(days=90)
DAYS = [START + timedelta(days=i) for i in range(91)]

# COMMAND ----------
# MAGIC %md ## Sites & equipment
# COMMAND ----------
sites = pd.DataFrame([
    ("PARA", "Paracatu", "Brazil", "Gold", 78.0),
    ("TASI", "Tasiast", "Mauritania", "Gold", 91.0),
    ("FKNX", "Fort Knox", "United States", "Gold", 84.0),
], columns=["site_id", "site_name", "country", "commodity", "recovery_target_pct"])

equip_rows = []
for sid in ["PARA", "TASI", "FKNX"]:
    equip_rows += [
        (f"{sid}-CRUSH-1", sid, "Primary Crusher", f"{sid} Primary Crusher"),
        (f"{sid}-SAG-1", sid, "SAG Mill", f"{sid} SAG Mill #1"),
        (f"{sid}-SAG-2", sid, "SAG Mill", f"{sid} SAG Mill #2"),
        (f"{sid}-BALL-1", sid, "Ball Mill", f"{sid} Ball Mill #1"),
    ]
equipment = pd.DataFrame(equip_rows, columns=["equipment_id", "site_id", "equipment_type", "equipment_name"])

CULPRIT = "PARA-SAG-2"   # the hero of the investigation

# COMMAND ----------
# MAGIC %md ## Daily production — the recovery dip
# COMMAND ----------
GOLD_PRICE = 2400.0  # USD/oz (synthetic, held flat)
site_cfg = {  # (base_recovery, target_grind_um, ore_tph, head_grade_gpt)
    "PARA": (78.0, 106.0, 4200.0, 0.44),
    "TASI": (91.0, 95.0, 1400.0, 1.75),
    "FKNX": (84.0, 120.0, 1300.0, 0.85),
}

prod = []
for sid, (base_rec, tgt_grind, ore_tph, grade) in site_cfg.items():
    for d in DAYS:
        days_ago = (TODAY - d).days
        # Paracatu grind coarsens over the final 10 days as SAG-2 degrades
        if sid == "PARA" and days_ago <= 10:
            ramp = (10 - days_ago) / 10.0          # 0 -> 1 toward today
            grind = tgt_grind + ramp * 46.0 + RNG.normal(0, 3)   # ~106 -> ~152 um
        else:
            grind = tgt_grind + RNG.normal(0, 4)
        # Recovery falls ~0.15 pp per um of grind above target
        recovery = base_rec - max(0.0, grind - tgt_grind) * 0.15 + RNG.normal(0, 0.4)
        ore = ore_tph * 24 * RNG.uniform(0.9, 1.05)
        hg = grade * RNG.uniform(0.92, 1.08)
        gold_oz = ore * hg / 31.1035 * recovery / 100.0
        prod.append((d, sid, round(ore, 1), round(hg, 3), round(ore / 24, 1),
                     round(grind, 1), round(recovery, 2), round(gold_oz, 1), GOLD_PRICE))

production_daily = pd.DataFrame(prod, columns=[
    "production_date", "site_id", "ore_tonnes", "head_grade_gpt", "throughput_tph",
    "grind_p80_um", "recovery_pct", "gold_oz", "gold_price_usd"])

# COMMAND ----------
# MAGIC %md ## Sensor readings — the vibration anomaly on the culprit
# COMMAND ----------
HOURS = [datetime.combine(START, datetime.min.time()) + timedelta(hours=h)
         for h in range(91 * 24)]
sensor_specs = {  # sensor_type -> (unit, base, noise)
    "vibration_mm_s": ("mm/s", 3.2, 0.35),
    "bearing_temp_c": ("degC", 62.0, 1.5),
    "power_kw": ("kW", 8200.0, 180.0),
}
sensor_rows = []
for eid in equipment["equipment_id"]:
    for stype, (unit, base, noise) in sensor_specs.items():
        for ts in HOURS:
            days_ago = (TODAY - ts.date()).days
            val = base + RNG.normal(0, noise)
            if eid == CULPRIT and days_ago <= 14 and stype in ("vibration_mm_s", "bearing_temp_c"):
                ramp = (14 - days_ago) / 14.0
                if stype == "vibration_mm_s":
                    val += ramp * 5.8      # 3.2 -> ~9.0 mm/s (alarm at 6.0)
                else:
                    val += ramp * 14.0     # bearing runs hot
            sensor_rows.append((ts, eid, stype, round(val, 2), unit))

sensor_readings = pd.DataFrame(sensor_rows, columns=[
    "reading_ts", "equipment_id", "sensor_type", "value", "unit"])

# COMMAND ----------
# MAGIC %md ## Work orders
# COMMAND ----------
wo_rows = []
wid = 1000
for sid in ["PARA", "TASI", "FKNX"]:
    for _ in range(12):
        wid += 1
        eid = RNG.choice(equipment[equipment.site_id == sid]["equipment_id"].values)
        created = START + timedelta(days=int(RNG.integers(0, 80)))
        wo_rows.append((f"WO-{wid}", eid, sid, created, "Preventive",
                        "Medium", "Completed", round(float(RNG.uniform(2, 8)), 1),
                        "Scheduled preventive maintenance"))
# the un-actioned high-priority order for the culprit
wid += 1
wo_rows.append((f"WO-{wid}", CULPRIT, "PARA", TODAY - timedelta(days=11), "Corrective",
                "High", "Open", 0.0,
                "Investigate elevated spindle bearing vibration on SAG Mill #2 flagged by condition monitoring"))
work_orders = pd.DataFrame(wo_rows, columns=[
    "work_order_id", "equipment_id", "site_id", "created_date", "wo_type",
    "priority", "status", "downtime_hours", "description"])

# COMMAND ----------
# MAGIC %md ## Condition monitoring — the ignored inspection
# COMMAND ----------
cm_rows = []
cid = 500
for eid in equipment["equipment_id"]:
    for wk in range(0, 90, 15):
        cid += 1
        insp = START + timedelta(days=wk)
        cm_rows.append((f"CM-{cid}", eid, insp, "Spindle bearing vibration",
                        round(float(RNG.uniform(2.8, 4.2)), 2), 6.0, "Normal",
                        "R. Okonkwo", "Within limits."))
# the ignored one
cid += 1
cm_rows.append(("CM-CRITICAL", CULPRIT, TODAY - timedelta(days=12), "Spindle bearing vibration",
                7.8, 6.0, "Acknowledged - No Action", "R. Okonkwo",
                "Vibration 7.8 mm/s exceeds 6.0 mm/s alarm threshold. Signature consistent with "
                "bearing wear. Recommend bearing inspection at next opportunity."))
condition_monitoring = pd.DataFrame(cm_rows, columns=[
    "cm_id", "equipment_id", "inspection_date", "parameter",
    "reading", "threshold", "status", "inspector", "notes"])

# COMMAND ----------
# MAGIC %md ## Write all tables to Delta
# COMMAND ----------
def write(df, name):
    (spark.createDataFrame(df)
        .write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{FQ}.{name}"))
    print(f"  wrote {FQ}.{name}: {len(df)} rows")

write(sites, "sites")
write(equipment, "equipment")
write(production_daily, "production_daily")
write(sensor_readings, "sensor_readings")
write(work_orders, "work_orders")
write(condition_monitoring, "condition_monitoring")

# COMMAND ----------
# MAGIC %md ## Sanity check — the dip is visible
# COMMAND ----------
display(spark.sql(f"""
  SELECT production_date, round(recovery_pct,1) recovery_pct, round(grind_p80_um,0) grind_p80_um
  FROM {FQ}.production_daily
  WHERE site_id='PARA' AND production_date >= date_sub(current_date(), 14)
  ORDER BY production_date
"""))
