# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the schema, a managed Volume for raw documents, and prints the config
# MAGIC every downstream notebook relies on. Safe to re-run (idempotent).

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
dbutils.widgets.text("warehouse_id", "a5246712a94a1cce")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
FQ = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------
# Catalog is provisioned already; do not CREATE it (account uses default storage,
# and CREATE CATALOG would require an explicit MANAGED LOCATION).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.raw_docs")
print(f"Schema ready:   {FQ}")
print(f"Volume ready:   /Volumes/{CATALOG}/{SCHEMA}/raw_docs")
print(f"Warehouse:      {WAREHOUSE_ID}")

# COMMAND ----------
# MAGIC %md
# MAGIC Config summary — downstream notebooks read the same three widgets.
# COMMAND ----------
print({"catalog": CATALOG, "schema": SCHEMA, "warehouse_id": WAREHOUSE_ID})
