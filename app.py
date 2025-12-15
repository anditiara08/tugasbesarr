import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# JUDUL APLIKASI
# =========================
st.title("Prediksi Loan Approval")
st.write("Classification menggunakan Random Forest")

# =========================
# UPLOAD DATASET
# =========================
st.subheader("Upload Dataset CSV")

uploaded_file = st.file_uploader(
    "Upload dataset Loan (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Silakan upload dataset CSV terlebih dahulu")
    st.stop()

data = pd.read_csv(uploaded_file)
data.columns = data.columns.str.strip()

st.success("Dataset berhasil di-upload")

# =========================
# PREVIEW DATA
# =========================
st.subheader("Preview Dataset")
st.dataframe(data.head())

# =========================
# CEK KOLOM WAJIB
# =========================
required_columns = [
    'loan_id',
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value',
    'loan_status'
]

if not all(col in data.columns for col in required_columns):
    st.error("Format dataset tidak sesuai dengan yang diminta")
    st.stop()

# =========================
# PREPROCESSING
# =========================
# Drop loan_id (tidak berpengaruh ke prediksi)
data.drop('loan_id', axis=1, inplace=True)

# Missing value
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# =========================
# ENCODING DATA KATEGORI
# =========================
le = LabelEncoder()

data['education'] = le.fit_transform(data['education'])
data['self_employed'] = le.fit_transform(data['self_employed'])
data['loan_status'] = le.fit_transform(data['loan_status'])
# Approved = 1, Rejected = 0 (biasanya)

# =========================
# FITUR & TARGET
# =========================
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
# TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
st.success("Model berhasil dilatih")

# =========================
# EVALUASI MODEL
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Evaluasi Model")
st.write(f"Accuracy: **{accuracy:.2f}**")

# =========================
# INPUT DATA BARU
# =========================
st.subheader("Input Data Pemohon")

no_of_dependents = st.number_input("Jumlah Tanggungan", min_value=0)
education = st.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Pendapatan Tahunan", min_value=0)
loan_amount = st.number_input("Jumlah Pinjaman", min_value=0)
loan_term = st.number_input("Jangka Waktu Pinjaman (tahun)", min_value=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Nilai Aset Residensial", min_value=0)
commercial_assets_value = st.number_input("Nilai Aset Komersial", min_value=0)
luxury_assets_value = st.number_input("Nilai Aset Mewah", min_value=0)
bank_asset_value = st.number_input("Nilai Aset Bank", min_value=0)

# Encoding input user
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi Loan"):
    input_data = pd.DataFrame([[
        no_of_dependents,
        education_val,
        self_employed_val,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan DISETUJUI")
    else:
        st.error("❌ Loan DITOLAK")
