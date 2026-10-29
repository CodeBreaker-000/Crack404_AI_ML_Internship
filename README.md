# AI/ML Internship Projects

This repository contains my weekly projects and practice tasks completed during the Crack404 AI/ML Internship.

The projects cover machine learning, model persistence, computer vision, and OCR-based document processing.

## Project Structure

```text
AI_ML_Intern/
│
├── week1/
├── week2_churn_project/
├── week3_ocr_project/
├── README.md
└── .gitignore
```

## Week 1 – Customer Churn Prediction

The first project focuses on predicting customer churn using a Random Forest classification model.

### Main Tasks

- Loaded and explored the Telco Customer Churn dataset
- Cleaned missing values in `TotalCharges`
- Converted categorical features into machine-readable format
- Split the dataset into training and testing sets
- Built a preprocessing and Random Forest pipeline
- Evaluated model performance using accuracy, recall, classification report, and confusion matrix
- Tested different probability thresholds to improve churn recall
- Analysed the most important features affecting churn prediction

### Main Files

```text
week1/
├── customer_churn_prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── requirements.txt
```

---

## Week 2 – Model Serialization and Standalone Prediction

The second project focuses on saving a trained machine-learning pipeline and using it later without retraining the model.

### Main Tasks

- Reused the trained customer churn pipeline
- Saved the model using Pickle and Joblib
- Compared model file formats
- Created a standalone Python prediction script
- Loaded the saved model without retraining
- Used hard-coded customer information for prediction
- Created a version-specific `requirements.txt` file

### Main Files

```text
week2_churn_project/
├── model_v1.joblib
├── model_v1.pkl
├── predict.py
├── Week_2.ipynb
└── requirements.txt
```

---

## Week 3 – Automated Receipt OCR Pipeline

The third project focuses on extracting text from receipt images using OpenCV and EasyOCR.

### Main Tasks

- Collected five receipt images
- Converted images to grayscale
- Applied Gaussian Blur for noise reduction
- Applied Global Thresholding using Otsu's method
- Applied Adaptive Thresholding
- Saved cleaned image versions
- Used EasyOCR to detect text
- Extracted OCR confidence scores and bounding-box coordinates
- Used a 60% working confidence threshold to flag uncertain results for human review
- Compared OCR performance on raw, global-thresholded, and adaptive-thresholded images
- Automatically selected the method with the highest average confidence for each receipt
- Generated bounding-box visualisations
- Saved structured OCR results to JSON
- Generated a text-based OCR analysis report

### Main Files

```text
week3_ocr_project/
├── raw_images/
├── cleaned_images/
│   ├── global/
│   └── adaptive/
├── visualized_images/
├── outputs/
│   ├── ocr_results.json
│   └── ocr_report.txt
├── ocr_pipeline.py
└── requirements.txt
```

---

## Installation

Each project contains its own `requirements.txt` file.

To install the required packages for a specific project, move into that project folder and run:

```bash
pip install -r requirements.txt
```

For example:

```bash
cd week3_ocr_project
pip install -r requirements.txt
```

## Running the Week 3 OCR Project

```bash
python ocr_pipeline.py
```

The script processes all receipt images inside `raw_images` and saves the results in the appropriate output folders.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- EasyOCR
- PyTorch
- Joblib
- Pickle

## Repository Purpose

This repository documents my practical learning and project development throughout the Crack404 AI/ML Internship, from structured machine-learning problems to model persistence and real-world OCR processing.