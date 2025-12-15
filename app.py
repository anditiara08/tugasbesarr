import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Loan Approval", layout="centered")
st.title("Aplikasi Prediksi Loan Approval")
st.write("Perbandingan Model Random Forest dan XGBoost")

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.lower()
    return data

data = load_data()

num_cols = data.select_dtypes(include=np.number).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].mean())

cat_cols = data.select_dtypes(include="object").columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

drop_cols = ["loan_status"]
if "loan_id" in data.columns:
    drop_cols.append("loan_id")

X = data.drop(columns=drop_cols)
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.subheader("Pilih Model")
model_choice = st.selectbox(
    "Model Classification",
    ["Random Forest", "XGBoost"]
)

MODEL_RF = "random_forest_model.pkl"
MODEL_XGB = "xgboost_model.pkl"

if model_choice == "Random Forest":
    if os.path.exists(MODEL_RF):
        model = joblib.load(MODEL_RF)
        st.success("Random Forest model dimuat.")
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_RF)
        st.success("Random Forest model dilatih.")

else:  # XGBoost
    if os.path.exists(MODEL_XGB):
        model = joblib.load(MODEL_XGB)
        st.success("XGBoost model dimuat.")
    else:
        st.error("File xgboost_model.pkl tidak ditemukan!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Akurasi Model")
st.write(f"Accuracy: **{accuracy:.2f}**"
st.subheader("Input Data Pemohon")

inputs = {}
for col in X.columns:
    if col in cat_cols:
        inputs[col] = st.selectbox(col, sorted(data[col].unique()))
    else:
        inputs[col] = st.number_input(col, value=float(data[col].mean()))

if st.button("Prediksi Loan"):
    input_df = pd.DataFrame([inputs])

    proba = model.predict_proba(input_df)[0][1]
    st.write(f"Peluang Disetujui: **{proba:.2%}**")

    if proba >= 0.35:
        st.success("✅ Loan DISETUJUI")
    else:
        st.error("❌ Loan DITOLAK")

if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Fitur": X.columns,
        "Pengaruh": model.feature_importances_
    }).sort_values(by="Pengaruh", ascending=False)
    st.dataframe(importance_df)
