import os, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))
client = MlflowClient()
name = "loan_default"

vers = client.search_model_versions(f"name='{name}'")
assert vers, "Aucune version trouvée pour 'loan_default' (relance l'entraînement)."
latest = max(vers, key=lambda v: int(v.version))

# Pose l'alias 'champion' sur la dernière version
client.set_registered_model_alias(name=name, alias="champion", version=latest.version)

mv = client.get_model_version_by_alias(name, "champion")
print(f"Alias 'champion' -> version {mv.version}")
