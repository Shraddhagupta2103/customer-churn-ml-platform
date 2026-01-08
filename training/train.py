import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess_data

# ---------- PATH SETUP ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")

# ---------- LOAD DATA ----------

DATA_PATH = os.path.join(
    BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
df = pd.read_csv(DATA_PATH)

# ---------- PREPROCESS ----------

X, y, preprocessor = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y

)

# ---------- PIPELINE (THIS IS WHERE IT GOES) ----------

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42))
])

# ---------- TRAIN ----------

pipeline.fit(X_train, y_train)

# ---------- EVALUATE ----------

preds = pipeline.predict(X_test)
f1 = f1_score(y_test, preds)

print(f"F1 Score: {f1:.4f}")

# ---------- SAVE MODEL ----------

joblib.dump(pipeline, MODEL_PATH)

print(f"✅ Model pipeline saved at: {MODEL_PATH}")

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, preds))




