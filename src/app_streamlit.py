

import os
from pathlib import Path
import shutil
import mlflow
from mlflow.tracking import MlflowClient
import streamlit as st
import numpy as np

st.write("🚀 Application chargée avec succès — interface en cours d’affichage…")

# ----------------------------
# Configuration MLflow
# ----------------------------
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
MODEL_NAME   = os.environ.get("MODEL_NAME", "loan_default")
MODEL_ALIAS  = os.environ.get("MODEL_ALIAS", "champion")
MODEL_DIR    = Path(__file__).parent / "model"  # modèle dans src/model/

# ----------------------------
# Fonction pour télécharger le modèle MLflow
# ----------------------------
def download_model():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    uri = client.get_model_version_download_uri(MODEL_NAME, mv.version)

    # Supprimer l'ancien modèle si présent
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    # Télécharger le modèle dans src/model/
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(MODEL_DIR))
    st.success(f"Modèle {MODEL_NAME}@{MODEL_ALIAS} téléchargé dans {MODEL_DIR} !")

# ----------------------------
# Charger le modèle MLflow
# --------------------
