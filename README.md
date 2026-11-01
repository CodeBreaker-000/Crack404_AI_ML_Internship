# Crack404 AI/ML Internship

This repository contains the practical projects I completed during my **Crack404 AI/ML Internship**.

The work covers machine learning, model persistence, computer vision, and OCR-based automation using Python.

---

## 📌 Projects Overview

| Week | Project | Main Focus | Status |
|---|---|---|---|
| Week 1 | Customer Churn Prediction | Machine Learning, Random Forest, Model Evaluation | Completed |
| Week 2 | Model Serialization & Inference | Pickle, Joblib, Standalone Prediction | Completed |
| Week 3 | Automated Receipt OCR | OpenCV, EasyOCR, Image Processing | Completed |
| Week 4 | Customer Sentiment Analysis | NLP, Text Processing, Deep Learning | In Progress |

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

The fourth project focuses on analysing customer sentiment from text using Natural Language Processing and Deep Learning.

### Work Completed So Far

- Set up the Week 4 project structure
- Loaded the IMDB Movie Reviews dataset
- Analysed the dataset structure and sentiment labels
- Checked for missing values
- Identified and removed duplicate reviews
- Reduced the dataset from **50,000 to 49,582 unique reviews**
- Analysed the final sentiment distribution:
  - Positive: **24,884 (50.19%)**
  - Negative: **24,698 (49.81%)**
- Saved the cleaned dataset as `IMDB_Cleaned.csv`
- Prepared the dataset for the upcoming text preprocessing and neural network stages

### Next Steps

- Clean and tokenize the review text
- Convert words into numerical sequences
- Apply sequence padding
- Build a neural network with an Embedding layer
- Train and evaluate the sentiment model
- Test the model using three custom sentences
- Save the trained model weights

### Technologies

`Python` `Pandas` `NumPy` `PyTorch` `NLP`

📁 [`week4_sentiment_analysis/`](./week4_sentiment_analysis)

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
- Week 4 — In Progress

More projects will be added as the internship continues.

---

## 👨‍💻 Author

**CodeBreaker-000**

GitHub: [CodeBreaker-000](https://github.com/CodeBreaker-000)

This repository documents my practical learning and project development throughout the Crack404 AI/ML Internship.