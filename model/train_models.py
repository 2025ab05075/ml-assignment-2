import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef


print("🚀 Training started...")

# ==========================
# Load Data
# ==========================
df = pd.read_csv("../data/bank-full.csv", sep=";")

# ==========================
# Encode Target
# ==========================
df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop("y", axis=1)
y = df["y"]

# ==========================
# Encode categorical columns
# ==========================
encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# SAVE encoders ⭐
joblib.dump(encoders, "../model/encoders.pkl")


# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================
# Scaling
# ==========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "../model/scaler.pkl")


# ==========================
# Models
# ==========================
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric='logloss')
}


# ==========================
# Train + Evaluate
# ==========================
results = []

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    res = [
        name,
        accuracy_score(y_test, preds),
        roc_auc_score(y_test, probs),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds),
        matthews_corrcoef(y_test, preds)
    ]

    results.append(res)

    # Save model
    joblib.dump(model, f"../model/{name}.pkl")


# ==========================
# Save Metrics
# ==========================
cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
pd.DataFrame(results, columns=cols).to_csv("../model/metrics.csv", index=False)

print("✅ Training completed & models saved!")
