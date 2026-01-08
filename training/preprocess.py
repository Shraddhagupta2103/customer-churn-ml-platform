import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    df = df.copy()

    # Target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fix TotalCharges (string → numeric)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna()

    num_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    cat_features = [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "PaperlessBilling"
    ]

    X = df[num_features + cat_features]
    y = df["Churn"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ]
    )

    return X, y, preprocessor
