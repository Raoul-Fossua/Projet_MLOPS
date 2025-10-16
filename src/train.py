from __future__ import annotations
import os, numpy as np, mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.synth_data import generate_apple_sales

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "Apple_Models")

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    data = generate_apple_sales(n_rows=2000)
    X = data.drop(columns=["date","demand"])
    y = data["demand"]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    params = dict(n_estimators=120, max_depth=6, min_samples_split=10, min_samples_leaf=4, random_state=888)
    with mlflow.start_run(run_name="apples_rf"):
        m = RandomForestRegressor(**params).fit(Xtr, ytr)
        yhat = m.predict(Xva)
        mse = mean_squared_error(yva, yhat)
        metrics = {"mae": mean_absolute_error(yva, yhat), "mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2_score(yva, yhat)}
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(m, artifact_path="rf_apples", input_example=Xva.head(3))
        print("metrics:", metrics)

if __name__ == "__main__":
    main()
