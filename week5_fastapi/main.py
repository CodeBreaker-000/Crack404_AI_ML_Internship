from contextlib import asynccontextmanager
from typing import Literal
import os

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


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
    gender: Literal["Male", "Female"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


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

    