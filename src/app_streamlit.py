# src/app_streamlit.py
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import streamlit as st
import numpy as np

# ----------------------------
# Config MLflow
# ----------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "loan_default")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
MODEL_DIR = Path(__file__).parent / "model"

# ----------------------------
# Télécharger le modèle MLflow
# ----------------------------
def download_model():
    from mlflow.tracking import MlflowClient
    import shutil

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    uri = client.get_model_version_download_uri(MODEL_NAME, mv.version)

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(MODEL_DIR))
    st.success(f"Modèle {MODEL_NAME}@{MODEL_ALIAS} téléchargé !")

# ----------------------------
# Charger le modèle
# ----------------------------
@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "model"  # "model" car MLflow log_model l'a nommé ainsi
    return mlflow.sklearn.load_model(str(model_path))

# ----------------------------
# Interface Streamlit
# ----------------------------
st.title("📊 Prédiction de défaut de prêt")

st.sidebar.header("Téléchargement du modèle")
if st.sidebar.button("Télécharger le modèle depuis MLflow"):
    download_model()

if MODEL_DIR.exists():
    model = load_model()
    st.sidebar.success("Modèle chargé !")

    st.header("Entrer les caractéristiques de l'emprunteur")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        income = st.number_input("Revenu annuel (€)", min_value=0, value=30000)
        loan_amount = st.number_input("Montant du prêt (€)", min_value=0, value=5000)
    with col2:
        credit_score = st.number_input("Score de crédit", min_value=300, max_value=850, value=650)
        previous_defaults = st.number_input("Prêts en défaut précédents", min_value=0, value=0)
        employment_years = st.number_input("Années d'emploi", min_value=0, value=3)

    if st.button("Prédire le risque de défaut"):
        X = np.array([[age, income, loan_amount, credit_score, previous_defaults, employment_years]])
        y_hat = model.predict(X)[0]
        st.subheader("Résultat de la prédiction :")
        st.success(f"Probabilité de défaut : {y_hat:.2f}")

else:
    st.warning("Le modèle n'a pas encore été téléchargé. Cliquez sur 'Télécharger le modèle' dans la barre latérale.")
