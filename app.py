import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Loan Approval", layout="centered")

st.title("Aplikasi Prediksi Loan Approval")
st.write("Model Classification menggunakan Random Forest")

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.lower()
    return data

data = load_data()

st.subheader("Preview Dataset")
st.dataframe(data.head())

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

st.write("Fitur yang dipakai model:")
st.write(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

MODEL_PATH = "random_forest_model.pkl"

retrain = True
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    # cek kecocokan fitur
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) == list(X.columns):
            retrain = False

if retrain:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.success("Model dilatih ulang (fitur konsisten).")
else:
    st.success("Model lama cocok, berhasil dimuat.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Akurasi Model")
st.write(f"Accuracy: **{accuracy:.2f}**")

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

st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Fitur": X.columns,
    "Pengaruh": model.feature_importances_
}).sort_values(by="Pengaruh", ascending=False)

st.dataframe(importance_df)
