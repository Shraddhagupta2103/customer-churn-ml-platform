# Customer Churn ML Platform

## 📌 Project Overview

This project is an **end-to-end Machine Learning system** for predicting **customer churn**. It simulates how ML models are built, deployed, monitored, and used in real business environments.

The goal is **not just model accuracy**, but to demonstrate **ML engineering best practices** such as reproducible training, real-time inference, batch inference, prediction logging, and data drift detection.

## 🧠 Problem Statement

Customer churn refers to customers leaving a service or subscription. Predicting churn allows businesses to:

* Identify high-risk customers early
* Run targeted retention campaigns
* Reduce revenue loss

This project predicts whether a customer is likely to churn based on historical usage and account information.

## 📊 Dataset

* **Source:** Telco Customer Churn Dataset
* **Rows:** ~7,000 customers
* **Target Variable:** `Churn` (Yes / No)

### Features Used

* `tenure`
* `MonthlyCharges`
* `TotalCharges`
* `Contract`
* `PaymentMethod`
* `InternetService`
* `TechSupport`
* `PaperlessBilling`

The `Churn` column is used **only during training** and is excluded during inference.

## ⚙️ System Architecture

```
Data (CSV)
   ↓
Preprocessing (scaling + encoding)
   ↓
Model Training (Pipeline)
   ↓
Saved Model Artifact (joblib)
   ↓
Inference (API / Batch)
   ↓
Prediction Logging
   ↓
Data Drift Detection
```

## 🤖 Model Training

* **Type:** Supervised Binary Classification
* **Pipeline Includes:**

  * Numeric scaling
  * Categorical encoding
  * Model training

Multiple models were evaluated. Several converged to similar performance, indicating a **feature-level performance ceiling**, which is common in churn prediction problems.

### Evaluation Metric

* **F1 Score ≈ 0.62**

This score is realistic for churn prediction and reflects a balanced trade-off between precision and recall.


## 🔍 Why F1 Score ≈ 0.62 is Acceptable

* Churn is noisy, behavior-driven, and imbalanced
* Many production churn models operate in the 0.60–0.70 F1 range
* The model shows **stable and interpretable behavior**
* Focus is on system reliability rather than metric over-optimization

## 🚀 Inference

### 1️⃣ Real-Time Inference (FastAPI)

* Endpoint: `POST /predict`
* Accepts customer details
* Returns:

  * Churn prediction (Yes / No)
  * Churn probability

This simulates how ML models are served in production.

### 2️⃣ Batch Inference

* Loads a CSV file of customers
* Predicts churn for **thousands of rows at once**
* Saves predictions to an output CSV

This simulates nightly or weekly batch scoring used in real businesses.

## 📝 Prediction Logging

Every inference request is logged with:

* Timestamp
* Input features
* Predicted label
* Predicted probability

Logs are stored in a CSV file and used for monitoring and drift detection.

## 📉 Data Drift Detection

* Compares training data vs inference data
* Uses **Kolmogorov–Smirnov (KS) Test** on numeric features
* Detects distribution shifts that may require retraining

This demonstrates **MLOps and production monitoring awareness**.

## 📁 Project Structure

```
customer_churn_ml_platform/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── training/
│   ├── preprocess.py
│   └── train.py
│
├── models/
│   └── churn_model.pkl
│
├── inference/
│   ├── __init__.py
│   ├── app.py
│   ├── schema.py
│   ├── batch_inference.py
│   └── churn_predictions_batch.csv
│
├── monitoring/
│   ├── predictions_log.csv
│   └── drift_check.py
│
├── README.md
└── requirements.txt

```

## ▶️ How to Run

### Train the Model

```bash
python training/train.py
```

### Start Inference API

```bash
python -m uvicorn inference.app:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

### Run Batch Inference

```bash
python inference/batch_inference.py
```

### Run Drift Detection

```bash
python monitoring/drift_check.py
```

## 💼 What This Project Demonstrates

✔ End-to-end ML lifecycle
✔ Clean separation of training and inference
✔ Production-style pipelines
✔ Real-time & batch prediction
✔ Logging & monitoring readiness
✔ MLOps and ML Architect thinking

## 📌 Future Improvements

* Automated retraining triggers
* Scheduling with workflow orchestration
* Dashboard for monitoring metrics
* Database-backed logging

## 🏁 Final Note

This project focuses on **engineering robustness and real-world applicability**, not leaderboard-style optimization. It reflects how ML systems are built and maintained in production environments.

**Author:** Shraddha Gupta
