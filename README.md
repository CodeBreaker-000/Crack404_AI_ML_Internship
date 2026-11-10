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
| Week 5 | Customer Churn Prediction API | FastAPI, Pydantic, Model Integration | In Progress |

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

The fifth project focuses on exposing the previously trained Customer Churn model through a FastAPI microservice so external applications can send customer information and receive predictions through an API.

### Work Completed So Far

- Created the Week 5 FastAPI project structure
- Installed and configured FastAPI and Uvicorn
- Reused the saved `model_v1.joblib` churn model from Week 2
- Implemented a FastAPI `lifespan` function to load the model once when the server starts
- Stored the loaded model in memory for reuse across prediction requests
- Defined a Pydantic `CustomerData` schema using the customer features required by the trained churn model
- Added appropriate data types for numerical and categorical customer information
- Created a root `/` endpoint to confirm that the API server is running
- Created a `POST /predict` endpoint for customer churn predictions
- Converted validated Pydantic input into a Pandas DataFrame before passing it to the saved machine learning pipeline
- Used `predict_proba()` to calculate the probability of customer churn
- Applied the existing `0.40` decision threshold from the Week 1 churn model
- Returned the churn result, probability and threshold as a JSON response
- Successfully tested the API through FastAPI Swagger UI
- Tested both churn and non-churn customer profiles
- Received `churn: true` with a probability of `0.87` for one test customer
- Received `churn: false` with a probability of `0.00` for another test customer
- Identified and resolved a Scikit-learn version mismatch by using `scikit-learn==1.6.1`
- Created a version-locked `requirements.txt` for the API environment

### Next Steps

- Test missing and invalid customer input through Swagger UI
- Review Pydantic validation and FastAPI error responses
- Test how the API handles an empty JSON payload
- Review synchronous and asynchronous request handling
- Capture the final successful Swagger UI prediction screenshot
- Prepare for the Week 5 technical viva

### Technologies

`Python` `FastAPI` `Uvicorn` `Pydantic` `Joblib` `Pandas` `Scikit-learn`

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
│   ├── main.py
│   ├── model_v1.joblib
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
- Week 5 — In Progress

More projects will be added as the internship continues.

---

## 👨‍💻 Author

**CodeBreaker-000**

GitHub: [CodeBreaker-000](https://github.com/CodeBreaker-000)

This repository documents my practical learning and project development throughout the Crack404 AI/ML Internship.