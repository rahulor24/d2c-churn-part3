# Part 3: Churn Prediction Model & Model Card

This folder contains the deliverables for Part 3 of the Capstone Project: **D2C Customer Churn Intelligence & Retention API**.

## Overview
The goal of this part is to build and evaluate a predictive model to identify customers likely to churn in the next 60 days. The model utilizes demographic, purchase RFM, customer support, and web logs features.

---

## Directory Structure
* `train_model.py`: Python script performing split-based training, threshold optimization, feature importances plotting, error analysis, and serialization.
* `churn_model.ipynb`: Jupyter notebook containing the interactive modeling walkthrough.
* `model.pkl`: Serialized XGBoost classification pipeline.
* `metrics.json`: JSON output containing evaluation metrics for the validation and test splits.
* `error_analysis.md`: Detailed audit of **10 misclassified customer IDs** (5 FPs, 5 FNs).
* `model_card.md`: Model Card documenting model types, performance, biases, and guidelines.
* `feature_importance.png`: Bar chart of the top feature importances.
* `requirements.txt`: Python package dependencies.

---

## Model & Preprocessing
* **Model Type:** XGBoost Classifier (selected as the best model, beating baseline Logistic Regression and Random Forest on AUC).
* **Preprocessing:** Numeric features are imputed with the median and standardized using `StandardScaler`. Categorical features are One-Hot Encoded.
* **Leakage Control:** All training is strictly partitioned using the pre-assigned `split` flag. Features do not contain any post-snapshot date orders.

---

## Running Instructions

### 1. Prerequisites
Ensure Python (3.9+) is installed. Install the dependencies by running:
```bash
pip install -r requirements.txt
```

### 2. Train Model
To execute the model training and output verification artifacts:
```bash
python train_model.py
```

### 3. Interactive Analysis
Open the Jupyter notebook `churn_model.ipynb` to view the step-by-step modeling flow:
```bash
jupyter notebook churn_model.ipynb
```
