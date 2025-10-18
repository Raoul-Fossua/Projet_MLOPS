# src/train_loan_default.py
from __future__ import annotations
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn

# ---------- Config ----------
DATA_PATH = Path("data/Loan_default.csv")  # adapte si nécessaire
TARGET = "loan_default"                     # <- adapte au vrai nom de la cible du CSV
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Pour respecter "Un modèle -> un experiment"
EXPERIMENTS = {
    "logreg": "Loan_Default__LogisticRegression",
    "rf":     "Loan_Default__RandomForest",
    "dt":     "Loan_Default__DecisionTree",
}

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Harmonise les noms de colonnes
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

def split_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' introuvable. Colonnes: {list(df.columns)}")

    y = df[target_col].astype(int)  # binaire 0/1
    X = df.drop(columns=[target_col])

    # Typage basique : numériques vs catégorielles
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    schema = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "feature_order": numeric_cols + categorical_cols,
        "target": target_col,
    }
    return X_train, X_test, y_train, y_test, schema

def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),   # with_mean=False pour compat sparse
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def evaluate(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics

def log_to_mlflow(model_name: str, clf, preprocess, params: Dict[str, Any], metrics: Dict[str, float], schema: Dict[str, Any]):
    # 1 modèle -> 1 experiment
    mlflow.set_experiment(EXPERIMENTS[model_name])

    with mlflow.start_run(run_name=f"{model_name}_baseline"):
        # Params
        mlflow.log_params(params)
        mlflow.log_dict(schema, "schema.json")

        # Metrics
        mlflow.log_metrics(metrics)

        # Pipeline entière (préprocess + modèle)
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ])

        # Enregistre la pipeline sklearn dans MLflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",  # deprecated alias "artifact_path", accepté; on peut utiliser name= aussi
            registered_model_name=None  # tu peux mettre "Loan_Default_Best" plus tard côté meilleur run
        )

        run_id = mlflow.active_run().info.run_id
        print(f"🏃 MLflow run: {mlflow.get_tracking_uri()} -> {run_id}")
        print(f"Set MLFLOW_MODEL_URI='runs:/{run_id}/model' pour servir ce modèle.")
        return run_id

def train_one(model_name: str, X_train, X_test, y_train, y_test, schema):
    preproc = build_preprocessor(schema["numeric"], schema["categorical"])

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE)
        params = {"max_iter": 200, "class_weight": "balanced"}
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        )
        params = {"n_estimators": 300, "max_depth": None, "class_weight": "balanced"}
    elif model_name == "dt":
        clf = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE, class_weight="balanced")
        params = {"max_depth": None, "class_weight": "balanced"}
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")

    # Build pipeline and fit
    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)
    y_proba = None
    # Si proba dispo
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    metrics = evaluate(y_test, y_pred, y_proba)
    run_id = log_to_mlflow(model_name, clf, preproc, params, metrics, schema)
    return model_name, metrics, run_id

def main():
    # Assure-toi que MLFLOW_TRACKING_URI est défini (ton serveur tourne déjà)
    tracking = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
    mlflow.set_tracking_uri(tracking)
    print(f"Tracking sur: {tracking}")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, schema = split_features(df, TARGET)

    results = []
    for name in ["logreg", "rf", "dt"]:
        print(f"\n=== {name} ===")
        r = train_one(name, X_train, X_test, y_train, y_test, schema)
        results.append(r)

    # Résumé
    print("\nRésumé des runs :")
    for name, metrics, run_id in results:
        print(f"- {name}: {json.dumps(metrics, indent=2)} | run_id={run_id}")

    # Option : choisir le meilleur au ROC_AUC (ou F1)
    best = max(results, key=lambda x: x[1].get("roc_auc", -1.0))
    print(f"\n🏆 Meilleur (selon ROC_AUC): {best[0]} -> run_id={best[2]}")
    print(f"Pour servir: set MLFLOW_MODEL_URI=runs:/{best[2]}/model")

if __name__ == "__main__":
    main()
