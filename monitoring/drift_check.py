import os
import pandas as pd
from scipy.stats import ks_2samp

# ---------- PATH SETUP ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA_PATH = os.path.join(
    BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

LOG_PATH = os.path.join(
    BASE_DIR, "monitoring", "predictions_log.csv"
)

# ---------- LOAD DATA ----------

train_df = pd.read_csv(TRAIN_DATA_PATH)
log_df = pd.read_csv(LOG_PATH)

# ---------- NUMERIC FEATURES TO MONITOR ----------

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# ---------- CLEAN TRAIN DATA ----------

train_df["TotalCharges"] = pd.to_numeric(
    train_df["TotalCharges"], errors="coerce"
)
train_df = train_df.dropna()

# ---------- DRIFT CHECK ----------

print("🔍 Data Drift Report\n")

for feature in numeric_features:
    stat, p_value = ks_2samp(
        train_df[feature],
        log_df[feature]
    )

    drift_detected = p_value < 0.05

    print(f"Feature: {feature}")
    print(f"  KS statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if drift_detected:
        print("  ⚠️ Drift detected")
    else:
        print("  ✅ No drift detected")

    print("-" * 40)
