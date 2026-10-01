# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Vibration anomaly-detection model
# MAGIC Trains an IsolationForest over equipment sensor signals (vibration, bearing temp,
# MAGIC power), wraps it in a pyfunc that returns an anomaly score + flag, registers it to
# MAGIC Unity Catalog, and serves it as `kinross-vibration-anomaly`. This is what *should*
# MAGIC have caught the SAG Mill #2 signature the inspection report flagged and nobody
# MAGIC actioned.

# COMMAND ----------
# MAGIC %pip install mlflow scikit-learn joblib pandas numpy -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
dbutils.widgets.text("catalog", "ramin_serverless_aws_catalog")
dbutils.widgets.text("schema", "kinross_mining")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
MODEL_NAME = f"{FQ}.anomaly_model"
ENDPOINT = "kinross-vibration-anomaly"

# COMMAND ----------
import mlflow, pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
FEATURES = ["vibration_mm_s", "bearing_temp_c", "power_kw"]

# Pivot sensor readings to one row per equipment-hour with the 3 signals as columns
sdf = spark.table(f"{FQ}.sensor_readings").toPandas()
wide = (sdf.pivot_table(index=["equipment_id", "reading_ts"], columns="sensor_type",
                        values="value", aggfunc="mean").reset_index().dropna(subset=FEATURES))
print("training rows:", len(wide))

# COMMAND ----------
# MAGIC %md ## Train + wrap in a pyfunc that returns score & flag
# COMMAND ----------
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(wide[FEATURES])

class AnomalyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.model = joblib.load(context.artifacts["iso"])
    def predict(self, context, model_input):
        import pandas as pd
        X = model_input[["vibration_mm_s", "bearing_temp_c", "power_kw"]]
        raw = self.model.decision_function(X)          # higher = more normal
        score = (-raw).round(4)                          # higher = more anomalous
        flag = (self.model.predict(X) == -1)
        return pd.DataFrame({"anomaly_score": score, "is_anomaly": flag})

import joblib, os
os.makedirs("/tmp/anom", exist_ok=True)
joblib.dump(iso, "/tmp/anom/iso.joblib")

sample = wide[FEATURES].head(5)
example_out = AnomalyModel();
sig = infer_signature(sample, pd.DataFrame({"anomaly_score": [0.0]*5, "is_anomaly": [False]*5}))

# COMMAND ----------
with mlflow.start_run(run_name="anomaly_model") as run:
    mlflow.pyfunc.log_model(
        name="model",
        python_model=AnomalyModel(),
        artifacts={"iso": "/tmp/anom/iso.joblib"},
        signature=sig,
        input_example=sample,
        pip_requirements=["scikit-learn", "joblib", "pandas", "numpy"],
        registered_model_name=MODEL_NAME)
    print("logged + registered", MODEL_NAME)

# COMMAND ----------
from mlflow.tracking import MlflowClient
c = MlflowClient(registry_uri="databricks-uc")
latest = max(int(v.version) for v in c.search_model_versions(f"name='{MODEL_NAME}'"))
c.set_registered_model_alias(MODEL_NAME, "champion", latest)
print("model version:", latest)

# COMMAND ----------
# MAGIC %md ## Sanity: score normal vs the SAG-2 anomaly signature
# COMMAND ----------
test = pd.DataFrame({"vibration_mm_s": [3.3, 9.0], "bearing_temp_c": [62.0, 76.0], "power_kw": [8200.0, 8200.0]})
loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
display(spark.createDataFrame(pd.concat([test, loaded.predict(test)], axis=1)))

# COMMAND ----------
# MAGIC %md ## Serve the model
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
w = WorkspaceClient()
served = ServedEntityInput(entity_name=MODEL_NAME, entity_version=str(latest),
                           scale_to_zero_enabled=True, workload_size="Small")
existing = [e.name for e in w.serving_endpoints.list()]
if ENDPOINT in existing:
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[served])
    print("updating endpoint", ENDPOINT)
else:
    w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served]))
    print("creating endpoint", ENDPOINT)
print("Endpoint provisioning started (poll for READY separately).")
