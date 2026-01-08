import os
import joblib
import pandas as pd

# ---------- PATH SETUP ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "inference", "churn_predictions_batch.csv"
)

# ---------- LOAD MODEL ----------

pipeline = joblib.load(MODEL_PATH)

# ---------- LOAD DATA ----------

df = pd.read_csv(INPUT_PATH)

# Keep only features used by the model
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "TechSupport",
    "PaperlessBilling"
]

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df[features]

# ---------- BATCH PREDICTION ----------

predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)[:, 1]

# ---------- SAVE RESULTS ----------

output_df = df.copy()
output_df["churn_prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
output_df["churn_probability"] = probabilities.round(3)

output_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Batch inference completed")
print(f"📄 Predictions saved at: {OUTPUT_PATH}")
print(f"🔢 Total customers scored: {len(output_df)}")
