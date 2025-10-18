import os
from typing import Any, Dict, List

import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Loan Default API")

MLFLOW_MODEL_URI = os.environ.get("MLFLOW_MODEL_URI", "models:/loan_default@champion")
_model = None

@app.on_event("startup")
def _load_model():
    global _model
    _model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)

@app.get("/health")
def health():
    return {"status": "ok", "model": _model is not None, "model_uri": MLFLOW_MODEL_URI}
expected_cols = None  # en haut du fichier, près de MLFLOW_MODEL_URI

@app.on_event("startup")
def _load_model():
    global _model, expected_cols
    _model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    # essayer de déduire la liste des colonnes attendues
    try:
        # si tu avais loggé un input_example, on peut le lire :
        example = _model._get_or_infer_signature().inputs.to_dict()  # fallback si dispo
    except Exception:
        example = None
    # sinon, charge 1 ligne du CSV pour référence (optionnel)
    expected_cols = None  # si tu veux garder simple, laisse None

@app.post("/predict")
def predict(payload: Dict[str, Any] | List[Dict[str, Any]]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rows = payload if isinstance(payload, list) else [payload]
    df = pd.DataFrame(rows)

    # exemple de garde-fou minimal (facultatif)
    # required = [...]  # liste explicite des colonnes si tu veux être strict
    # missing = [c for c in required if c not in df.columns]
    # if missing:
    #     raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {"result": [{"prob_default": float(p), "prediction": int(y)} for p, y in zip(proba, pred)]}
    else:
        pred = _model.predict(df)
        return {"result": [{"prediction": int(y)} for y in pred]}

@app.post("/predict")
def predict(payload: Dict[str, Any] | List[Dict[str, Any]]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = payload if isinstance(payload, list) else [payload]
    df = pd.DataFrame(rows)

    # essaie probas si dispo, sinon classe directement
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out = [{"prob_default": float(p), "prediction": int(y)} for p, y in zip(proba, pred)]
    else:
        pred = _model.predict(df)
        out = [{"prediction": int(y)} for y in pred]

    return {"result": out}
