import joblib
import pandas as pd


# Load the trained model
model = joblib.load("model_v1.joblib")

print("Model loaded successfully.")


# Fake new customer data
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 1026.00
}])


# Make prediction
prediction = model.predict(new_customer)


# Display result
if prediction[0] == 1:
    print("Prediction: Customer is likely to CHURN.")
else:
    print("Prediction: Customer is likely to STAY.")