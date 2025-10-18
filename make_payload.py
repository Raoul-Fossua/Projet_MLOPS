import os, json, pandas as pd
csv = os.environ.get("LOAN_DATA_PATH", "Loan_default.csv")
df = pd.read_csv(csv)
drop = [c for c in ("LoanID","Default") if c in df.columns]
X = df.drop(columns=drop)
X.iloc[0:3].to_json("payload.json", orient="records")
print("payload.json prêt.")
