import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA_PATH = os.environ.get("LOAN_DATA_PATH", "Loan_default.csv")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "Loan_Default")
MODEL_NAME = "loan_default"
SEED = 42

# ---------- Data ----------
df = pd.read_csv(DATA_PATH)

target = "Default"
drop_cols = [c for c in ["LoanID", target] if c in df.columns]
y = df[target].astype(int)
X = df.drop(columns=drop_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preproc = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

candidates = {
    "logreg": LogisticRegression(max_iter=500, random_state=SEED),
    "rf": RandomForestClassifier(n_estimators=300, random_state=SEED),
}

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))
mlflow.set_experiment(EXPERIMENT)

best_auc = -1.0
best_run_id = None
best_name = None

for name, clf in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        pipe = Pipeline([("preproc", preproc), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        mlflow.log_metrics({"auc": float(auc), "f1": float(f1), "accuracy": float(acc),
                            "precision": float(prec), "recall": float(rec)})

        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion matrix - {name}")
        plt.colorbar()
        ticks = np.arange(2)
        plt.xticks(ticks, ["NoDefault", "Default"])
        plt.yticks(ticks, ["NoDefault", "Default"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.xlabel("Predicted"); plt.ylabel("True")
        mlflow.log_figure(fig, f"confusion_matrix_{name}.png")
        plt.close(fig)

        example = X_train.iloc[:3].copy()
        signature = infer_signature(example, pipe.predict_proba(example)[:, 1])

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=example,
            signature=signature,
        )

        run_id = run.info.run_id
        print(f"[{name}] AUC={auc:.4f}  run_id={run_id}")

        if auc > best_auc:
            best_auc = auc
            best_run_id = run_id
            best_name = name

client = MlflowClient()
mv = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{best_run_id}'")[0]
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=mv.version)

print(f"BEST_MODEL={best_name} AUC={best_auc:.4f} RUN_ID={best_run_id}")
print("Use URI: models:/loan_default@champion")
