from pydantic import BaseModel

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str
    TechSupport: str
    PaperlessBilling: str

class ChurnPredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
