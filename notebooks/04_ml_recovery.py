# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Recovery-prediction model
# MAGIC Trains a model that predicts **gold recovery %** from feed grade, grind size and
# MAGIC throughput, registers it to Unity Catalog, and serves it as
# MAGIC `kinross-recovery-prediction`. Genie-1 calls this to quantify the recovery gap and
# MAGIC the ounces/$ at stake ("at spec grind you'd recover ~78%, at current grind ~71%").

# COMMAND ----------
# MAGIC %pip install mlflow scikit-learn pandas -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{FQ}.recovery_model"
ENDPOINT = "kinross-recovery-prediction"

# COMMAND ----------
import mlflow, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

df = spark.table(f"{FQ}.production_daily").toPandas()
FEATURES = ["head_grade_gpt", "grind_p80_um", "throughput_tph"]
X, y = df[FEATURES], df["recovery_pct"]

# Recovery vs grind is essentially linear; a linear model fits well and serializes
# cleanly (no tree types), and still shows the grind -> recovery gap for the demo.
pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
pipe.fit(X, y)
print("R^2 on training data:", round(pipe.score(X, y), 4))

# COMMAND ----------
with mlflow.start_run(run_name="recovery_model") as run:
    sig = infer_signature(X, pipe.predict(X))
    mlflow.sklearn.log_model(pipe, name="model", signature=sig,
                             input_example=X.head(3), registered_model_name=MODEL_NAME)
    print("logged + registered", MODEL_NAME)

# COMMAND ----------
# MAGIC %md ## Latest registered version
# COMMAND ----------
from mlflow.tracking import MlflowClient
c = MlflowClient(registry_uri="databricks-uc")
versions = c.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print("model version:", latest)
c.set_registered_model_alias(MODEL_NAME, "champion", latest)

# COMMAND ----------
# MAGIC %md ## Sanity: predict recovery at spec vs current Paracatu grind
# COMMAND ----------
demo = pd.DataFrame({"head_grade_gpt": [0.44, 0.44], "grind_p80_um": [106.0, 150.0],
                     "throughput_tph": [4200.0, 4200.0]})
demo["pred_recovery_pct"] = pipe.predict(demo).round(2)
display(spark.createDataFrame(demo))

# COMMAND ----------
# MAGIC %md ## Serve the model (create or update endpoint)
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
w = WorkspaceClient()
served = ServedEntityInput(entity_name=MODEL_NAME, entity_version=str(latest),
                           scale_to_zero_enabled=True, workload_size="Small")
cfg = EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served])
existing = [e.name for e in w.serving_endpoints.list()]
if ENDPOINT in existing:
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[served])
    print("updating endpoint", ENDPOINT)
else:
    w.serving_endpoints.create(name=ENDPOINT, config=cfg)
    print("creating endpoint", ENDPOINT)
print("Endpoint provisioning started (poll for READY separately).")
