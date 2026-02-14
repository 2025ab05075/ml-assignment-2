import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


st.title("Bank Marketing Prediction App")

# ==========================
# Load Metrics from training
# ==========================
metrics = pd.read_csv("model/metrics.csv")
st.subheader("Model Performance (Training Data)")
st.dataframe(metrics)


# ==========================
# Choose Model
# ==========================
model_name = st.selectbox("Select Model", metrics["Model"].tolist())

model = joblib.load(f"model/{model_name}.pkl")
scaler = joblib.load("model/scaler.pkl")


# ==========================
# Upload CSV
# ==========================
st.subheader("Upload Test Dataset")
file = st.file_uploader("Upload CSV", type=["csv"])


if file is not None:
    df = pd.read_csv(file, sep=";")

    st.write("### Data Preview")
    st.dataframe(df.head())

    # default
    y_true = None

    if "y" in df.columns:
        y_true = df["y"].map({"yes": 1, "no": 0})
        X = df.drop("y", axis=1)
    else:
        X = df

    # encode categorical
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype('category').cat.codes

    # scale
    X = scaler.transform(X)

    # predict
    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(preds)


    # ==========================
    # Evaluation metrics (ONLY if y exists)
    # ==========================
    if y_true is not None:

        st.subheader("📊 Evaluation Metrics (Test Data)")

        probs = model.predict_proba(X)[:, 1]

        results = {
            "Accuracy": accuracy_score(y_true, preds),
            "AUC": roc_auc_score(y_true, probs),
            "Precision": precision_score(y_true, preds),
            "Recall": recall_score(y_true, preds),
            "F1 Score": f1_score(y_true, preds),
            "MCC": matthews_corrcoef(y_true, preds),
        }

        result_df = pd.DataFrame(
            {"Metric": results.keys(), "Value": results.values()}
        )

        st.dataframe(result_df, use_container_width=True)


        # ==========================
        # Confusion Matrix (still required)
        # ==========================
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, preds))
