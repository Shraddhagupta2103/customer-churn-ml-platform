import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from inference.schema import CustomerData, ChurnPredictionResponse
from datetime import datetime
import csv



# ---------- PATH SETUP ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "predictions_log.csv")


# ---------- LOAD MODEL PIPELINE ----------

pipeline = joblib.load(MODEL_PATH)

# ---------- FASTAPI APP ----------

app = FastAPI(title="Customer Churn Inference API")

# ---------- INPUT SCHEMA ----------

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str
    TechSupport: str
    PaperlessBilling: str

# ---------- OUTPUT ----------

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(data: CustomerData):

    input_df = pd.DataFrame([data.dict()])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    result = {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 3)
    }

    # -------- LOG PREDICTION --------
    with open(LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.utcnow().isoformat(),
            data.tenure,
            data.MonthlyCharges,
            data.TotalCharges,
            data.Contract,
            data.PaymentMethod,
            data.InternetService,
            data.TechSupport,
            data.PaperlessBilling,
            result["churn_prediction"],
            result["churn_probability"]
        ])

    return result
