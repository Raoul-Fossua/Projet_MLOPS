import os, shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
NAME     = os.environ.get("MODEL_NAME", "loan_default")
ALIAS    = os.environ.get("MODEL_ALIAS", "champion")
DEST     = Path("model")

def main():
    client = MlflowClient(tracking_uri=TRACKING)
    mv = client.get_model_version_by_alias(NAME, ALIAS)
    uri = client.get_model_version_download_uri(NAME, mv.version)

    if DEST.exists():
        shutil.rmtree(DEST)

    print(f"Downloading {NAME}@{ALIAS} (v{mv.version}) from {uri} -> {DEST}/")
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(DEST))
    print("Done.")

if __name__ == "__main__":
    main()
