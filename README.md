ml-assignment-2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- data/
│-- model/

# Machine Learning Assignment 2  
## Classification Models & Streamlit Deployment

---

## Problem Statement
The objective of this assignment is to build multiple machine learning classification models to predict whether a customer will subscribe to a term deposit.  
The project also demonstrates end-to-end deployment by creating an interactive Streamlit web application where users can upload new data and obtain predictions.

---

## Dataset Description
Dataset used: **Bank Marketing Dataset (UCI Repository)**

The dataset contains marketing campaign information of a Portuguese banking institution.  
The goal is to predict if the client will subscribe to a term deposit.

**Key details:**
- Number of instances: ~45,000  
- Number of input features: 16  
- Target column: `y`
  - yes → subscribed  
  - no → not subscribed  

The dataset contains both numerical and categorical attributes such as age, job, marital status, balance, loan information, and previous contact outcomes.

---

## Models Used
The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

   ## Observations on Model Performance

| Model | Observation |
|------|-------------|
| Logistic Regression | Provides stable baseline performance but assumes linear decision boundary. |
| Decision Tree | Easy to interpret but may overfit the data. |
| kNN | Performance depends heavily on scaling and choice of neighbors. |
| Naive Bayes | Fast and simple, but assumptions may reduce predictive power. |
| Random Forest | Strong performance due to ensemble averaging and reduced variance. |
| XGBoost | Achieved high AUC and strong generalization, often among the best performers. |

---

## Streamlit Application Features

The deployed application includes:

- Upload test dataset (CSV)  
- Select model from dropdown  
- Display model performance  
- Generate predictions  
- Show confusion matrix  
- Compute evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)

## Data to upload 
data to upload is found in main/data/test.csv

