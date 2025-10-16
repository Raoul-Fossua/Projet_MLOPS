from __future__ import annotations
import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MLOps API", version="0.1.0")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
MODEL_URI = os.getenv("MLFLOW_MODEL_URI")  # runs:/<run_id>/rf_apples
_model = None

class Features(BaseModel):
    average_temperature: float = Field(..., ge=-50, le=60)
    rainfall: float = Field(..., ge=0)
    weekend: int = Field(..., ge=0, le=1)
    holiday: int = Field(..., ge=0, le=1)
    price_per_kg: float = Field(..., ge=0)
    previous_days_demand: float = Field(..., ge=0)

def get_model():
    global _model
    if _model is None:
        if not MODEL_URI:
            raise RuntimeError("MLFLOW_MODEL_URI manquant")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
def predict(f: Features):
    try:
        m = get_model()
    except Exception as e:
        # modèle pas encore dispo → 503
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")
    # à adapter à ton vrai modèle : ici on suppose m.predict(X)
    import numpy as np
    x = np.array([[f.average_temperature, f.rainfall, f.weekend, f.holiday, f.price_per_kg, f.previous_days_demand]])
    y_hat = float(m.predict(x)[0])
    return {"prediction": y_hat}
