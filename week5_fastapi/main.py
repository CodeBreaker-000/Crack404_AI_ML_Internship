from contextlib import asynccontextmanager
import os

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


models = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_v1.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading churn model...")

    models["churn_model"] = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    yield

    models.clear()

    print("Model removed from memory.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Week 5 Crack404 AI/ML Internship Project",
    version="1.0.0",
    lifespan=lifespan
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float



@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }
    
    
@app.post("/predict")
def predict_churn(customer: CustomerData):

    customer_data = customer.model_dump()

    customer_df = pd.DataFrame([customer_data])

    model = models["churn_model"]

    churn_probability = model.predict_proba(customer_df)[0][1]

    threshold = 0.40

    churn_prediction = churn_probability >= threshold

    return {
        "churn": bool(churn_prediction),
        "churn_probability": round(float(churn_probability), 4),
        "threshold": threshold
    }

