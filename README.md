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

   	Model	Accuracy	AUC	Precision	Recall	F1	MCC
0	logistic_regression	0.891407719	0.87258269	0.594527363	0.225897921	0.32739726	0.320453221
1	decision_tree	0.878137786	0.705100833	0.479206049	0.479206049	0.479206049	0.410201666
2	knn	0.892292381	0.808924164	0.571672355	0.316635161	0.407542579	0.372370016
3	naive_bayes	0.83799624	0.812745957	0.355366027	0.472589792	0.405679513	0.318344191
4	random_forest	0.905230565	0.926134304	0.64844904	0.414933837	0.506051873	0.470361828
5	xgboost	0.905783479	0.926749115	0.628109453	0.47731569	0.542427497	0.496752242
<img width="802" height="169" alt="image" src="https://github.com/user-attachments/assets/dbc222ec-0dac-4d52-ba17-bc2cc2aed4ad" />


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


