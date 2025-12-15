import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# JUDUL APLIKASI
# =========================
st.title("Aplikasi Prediksi Loan Approval")
st.write("Klasifikasi menggunakan Random Forest")

# =========================
# UPLOAD DATASET
# =========================
st.subheader("Upload Dataset CSV")

uploaded_file = st.file_uploader(
    "Upload dataset Loan Approval (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()
    st.success("Dataset berhasil di-upload")
else:
    st.warning("Silakan upload dataset CSV terlebih dahulu")
    st.stop()

# =========================
# PREVIEW DATASET
# =========================
st.subheader("Preview Dataset")
st.dataframe(data.head())

# =========================
# PREPROCESSING
# =========================
# Missing value numerik
if 'LoanAmount' in data.columns:
    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)

if 'Loan_Amount_Term' in data.columns:
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)

if 'Credit_History' in data.columns:
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# Missing value kategorikal
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)

# =========================
# ENCODING DATA KATEGORIKAL
# =========================
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# =========================
# FITUR & TARGET
# =========================
if 'loan_status' not in data.columns:
    st.error("Kolom 'loan_status' tidak ditemukan dalam dataset")
    st.stop()

X = data.drop('loan_status', axis=1)
y = data['loan_status']

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# LOAD / TRAIN MODEL
# =========================
MODEL_PATH = "random_forest_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model berhasil dimuat dari file")
except:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    st.success("Model berhasil dilatih dan disimpan")

# =========================
# EVALUASI MODEL
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Evaluasi Model")
st.write(f"Accuracy: **{accuracy:.2f}**")

# =========================
# INPUT USER
# =========================
st.subheader("Input Data Pemohon")

inputs = {}
for col in X.columns:
    if data[col].nunique() <= 5:
        inputs[col] = st.selectbox(
            col,
            sorted(data[col].unique())
        )
    else:
        inputs[col] = st.number_input(
            col,
            value=float(data[col].mean())
        )

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi Loan"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan DISETUJUI")
    else:
        st.error("❌ Loan DITOLAK")
