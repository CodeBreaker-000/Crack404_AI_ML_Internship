# Crack404 AI/ML Internship

This repository contains the practical projects I completed during my **Crack404 AI/ML Internship**.

The work covers machine learning, model persistence, computer vision, OCR-based automation, Natural Language Processing, and Deep Learning using Python.

---

## 📌 Projects Overview

| Week | Project | Main Focus | Status |
|---|---|---|---|
| Week 1 | Customer Churn Prediction | Machine Learning, Random Forest, Model Evaluation | Completed |
| Week 2 | Model Serialization & Inference | Pickle, Joblib, Standalone Prediction | Completed |
| Week 3 | Automated Receipt OCR | OpenCV, EasyOCR, Image Processing | Completed |
| Week 4 | Customer Sentiment Analysis | NLP, PyTorch, Deep Learning | Completed |
| Week 5 | Customer Churn Prediction API | FastAPI, Pydantic, API Integration | Completed |

---

## Week 1 — Customer Churn Prediction

The first project focused on predicting customer churn using a Random Forest classification model.

### Main Work

- Cleaned and prepared the Telco Customer Churn dataset
- Handled missing values in `TotalCharges`
- Encoded categorical features
- Built a preprocessing and Random Forest pipeline
- Evaluated the model using accuracy, recall, classification report, and confusion matrix
- Tested different probability thresholds to improve churn recall
- Analysed the most important features influencing the model

### Technologies

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`

📁 [`week1/`](./week1)

---

## Week 2 — Model Serialization & Standalone Prediction

The second project focused on saving a trained machine learning pipeline and using it later without retraining.

### Main Work

- Saved the trained model using Pickle and Joblib
- Compared the serialized model files
- Created a standalone `predict.py` script
- Loaded the saved model for new customer predictions
- Used hard-coded customer data for inference
- Maintained project dependencies using `requirements.txt`

### Technologies

`Python` `Scikit-learn` `Joblib` `Pickle` `Pandas`

📁 [`week2_churn_project/`](./week2_churn_project)

---

## Week 3 — Automated Receipt OCR Pipeline

The third project focused on building an OCR pipeline for receipt and invoice images using OpenCV and EasyOCR.

### Main Work

- Collected and processed five receipt images
- Converted images to grayscale
- Applied Gaussian Blur for noise reduction
- Applied Global and Adaptive Thresholding
- Used EasyOCR for text extraction
- Extracted confidence scores and bounding-box coordinates
- Used a 60% working confidence threshold to flag uncertain results for human review
- Compared OCR results from raw, globally thresholded, and adaptively thresholded images
- Automatically selected the method with the highest average confidence for each receipt
- Generated bounding-box visualisations
- Saved structured results in JSON and text report formats

### Technologies

`Python` `OpenCV` `EasyOCR` `NumPy` `PyTorch`

📁 [`week3_ocr_project/`](./week3_ocr_project)

---

## Week 4 — Customer Sentiment Analysis

The fourth project focused on analysing customer sentiment from text using Natural Language Processing and Deep Learning.

### Main Work

- Loaded and analysed the IMDB Movie Reviews dataset
- Checked for missing values and duplicate reviews
- Removed duplicate records, reducing the dataset from **50,000 to 49,582 unique reviews**
- Analysed the final sentiment distribution:
  - Positive: **24,884 (50.19%)**
  - Negative: **24,698 (49.81%)**
- Converted sentiment labels into binary values
- Cleaned review text by removing HTML tags, unnecessary characters, and extra spaces
- Tokenized reviews into individual words
- Removed stopwords while keeping sentiment-important words such as `not`, `no`, and `never`
- Applied stemming to simplify word variations
- Created a vocabulary of the most frequent words
- Converted review text into numerical sequences
- Applied sequence padding to create equal-length model inputs
- Split the dataset into training and testing sets
- Converted the data into PyTorch tensors and DataLoaders
- Built a sentiment classification neural network using an Embedding layer and fully connected layers
- Used ReLU activation and Sigmoid output for binary classification
- Trained and evaluated the model using unseen test data
- Generated a classification report and confusion matrix
- Saved the trained model as `sentiment_model.pt`
- Saved the vocabulary for future predictions
- Tested the model using three completely new custom sentences

### Technologies

`Python` `Pandas` `NumPy` `NLTK` `Scikit-learn` `PyTorch` `NLP` `Deep Learning`

📁 [`week4_sentiment_analysis/`](./week4_sentiment_analysis)

---

## Week 5 — Customer Churn Prediction API

The fifth project focused on deploying the previously trained Customer Churn model through a FastAPI microservice. The API allows external applications to submit customer information, validate the input, and receive real-time churn predictions as JSON responses without retraining the model.

### Main Work

- Created a FastAPI application for customer churn prediction
- Reused the saved `model_v1.joblib` machine learning pipeline from Week 2
- Implemented FastAPI `lifespan` to load the trained model once during application startup
- Stored the loaded model in memory for reuse across prediction requests
- Defined a Pydantic `CustomerData` schema containing all 19 required customer features
- Applied strict categorical validation using `Literal` and numerical constraints using `Field`
- Created a root `GET /` endpoint to check API availability
- Implemented a `POST /predict` endpoint to process customer data and generate predictions
- Converted validated customer inputs into a Pandas DataFrame before passing them to the trained pipeline
- Used `predict_proba()` to calculate customer churn probabilities
- Applied the existing `0.40` decision threshold to classify customers as churn or non-churn
- Returned the churn prediction, probability, and threshold in JSON format
- Successfully tested the API through FastAPI's interactive Swagger UI
- Verified predictions for both churn and non-churn customer profiles
- Tested invalid inputs, including missing fields, incorrect data types, unsupported categories, and negative numerical values
- Confirmed that FastAPI returns HTTP `422` validation responses for invalid requests
- Used a synchronous `def` endpoint for machine learning inference to avoid blocking the asynchronous event loop
- Integrated FastAPI `BackgroundTasks` to record prediction results after API responses
- Verified successful background logging in `prediction_log.txt`
- Resolved a Scikit-learn version compatibility issue when loading the serialized model
- Maintained a version-locked `requirements.txt` for environment consistency

### API Testing Results

The prediction endpoint was successfully tested using two different customer profiles.

| Test Case | Churn Prediction | Probability | HTTP Status |
|---|---|---|---|
| Customer 1 | True | 87% | 200 |
| Customer 2 | False | 0% | 200 |

Both predictions used the `0.40` decision threshold. Additional validation tests confirmed that invalid customer data was rejected before reaching the trained machine learning model.

### Technologies

`Python` `FastAPI` `Uvicorn` `Pydantic` `Pandas` `Scikit-learn` `Joblib`

### Running the API

Install the required dependencies:

```bash
cd week5_fastapi
python -m pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn main:app --reload
```

Open the interactive Swagger UI documentation:

`http://127.0.0.1:8000/docs`

Use the `POST /predict` endpoint to submit customer details and receive a churn prediction.

### Project Structure

```text
week5_fastapi/
├── main.py
├── model_v1.joblib
├── requirements.txt
└── screenshots/
    └── swagger_prediction.png
```

The API also generates a local `prediction_log.txt` file to record prediction results through background tasks.

### Key Learning

This project provided practical experience in connecting a trained machine learning model to a web application using FastAPI. It covered API development, request validation, model persistence, prediction handling, background tasks, and the importance of managing synchronous and asynchronous operations.

📁 [`week5_fastapi/`](./week5_fastapi)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Pickle
- OpenCV
- EasyOCR
- PyTorch
- NLP
- NLTK
- Deep Learning
- FastAPI
- Uvicorn
- Pydantic

---

## Repository Structure

```text
Crack404_AI_ML_Internship/
│
├── week1/
│   ├── customer_churn_prediction.ipynb
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── requirements.txt
│
├── week2_churn_project/
│   ├── model_v1.joblib
│   ├── model_v1.pkl
│   ├── predict.py
│   ├── Week_2.ipynb
│   └── requirements.txt
│
├── week3_ocr_project/
│   ├── raw_images/
│   ├── cleaned_images/
│   ├── visualized_images/
│   ├── outputs/
│   ├── ocr_pipeline.py
│   └── requirements.txt
│
├── week3_ocr_project/
│   ├── data/
│   ├── models/
│   ├── outputs/
│   ├── screenshots/
│   ├── requirements.txt
│   └── sentiment_analysis.ipynb
│
│── week4_sentiment_analysis/
│   ├── data/
│   ├── models/
│   │   ├── sentiment_model.pt
│   │   └── vocabulary.json
│   ├── outputs/
│   │   └── IMDB_Cleaned.csv
│   ├── screenshots/
│   ├── sentiment_analysis.ipynb
│   └── requirements.txt
│
│── week5_fastapi/
│   ├── screenshots/
│   │   └── swagger_prediction.png
│   ├── main.py
│   ├── model_v1.joblib
│   ├── prediction_log.txt
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

---

## Installation

Each project contains its own `requirements.txt` file.

For example:

```bash
cd week3_ocr_project
pip install -r requirements.txt
```

---

## Running the Week 3 OCR Project

```bash
cd week3_ocr_project
python ocr_pipeline.py
```

The script processes the receipt images, compares different OCR input methods, generates visualised outputs, and saves the extracted results in the output folder.

---

## Current Progress

- Week 1 — Completed
- Week 2 — Completed
- Week 3 — Completed
- Week 4 — Completed
- Week 5 — Completed

More projects will be added as the internship continues.

---

## 👨‍💻 Author

**CodeBreaker-000**

GitHub: [CodeBreaker-000](https://github.com/CodeBreaker-000)

This repository documents my practical learning and project development throughout the Crack404 AI/ML Internship.