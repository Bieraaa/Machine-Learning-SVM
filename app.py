import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon="🫀", layout="centered")

st.title("🫀 Sistem Prediksi Penyakit Kardiovaskular")
st.write("Aplikasi ini menggunakan model Machine Learning (Random Forest) untuk memprediksi risiko penyakit jantung berdasarkan data rekam medis pasien.")

# 2. Load Model Terbaik (Gunakan Random Forest karena hasilnya paling realistis)
@st.cache_resource
def load_model():
    # Pastikan file .joblib berada di folder yang sama dengan app.py
    return joblib.load('model_rf_jantung.joblib')

model = load_model()

# 3. Membuat Form Input di Sidebar
st.sidebar.header("📝 Masukkan Data Pasien")

def user_input_features():
    age = st.sidebar.slider("Umur (Age)", 20, 100, 50)
    sex = st.sidebar.selectbox("Jenis Kelamin (Sex)", ("M", "F"))
    chest_pain = st.sidebar.selectbox("Tipe Nyeri Dada (ChestPainType)", ("ATA", "NAP", "ASY", "TA"))
    resting_bp = st.sidebar.slider("Tekanan Darah (RestingBP)", 80, 200, 120)
    cholesterol = st.sidebar.slider("Kolesterol (Cholesterol)", 100, 400, 200)
    resting_ecg = st.sidebar.selectbox("Hasil EKG (RestingECG)", ("Normal", "ST", "LVH"))
    max_hr = st.sidebar.slider("Detak Jantung Maksimal (MaxHR)", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Nyeri Dada saat Olahraga? (ExerciseAngina)", ("Y", "N"))
    oldpeak = st.sidebar.slider("Depresi ST (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.sidebar.selectbox("Kemiringan ST (ST_Slope)", ("Up", "Flat", "Down"))

    # Menyusun input menjadi DataFrame (Nama kolom WAJIB sama persis dengan dataset asli)
    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Menampilkan Data Inputan
st.subheader("Data Pasien Saat Ini")
st.write(input_df)

# 5. Tombol Prediksi
if st.button("Lakukan Prediksi", type="primary"):
    # Melakukan prediksi menggunakan pipeline
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Hasil Prediksi")
    
    if prediction[0] == 1:
        st.error("⚠️ **BERISIKO TINGGI** mengidap Penyakit Kardiovaskular.")
        st.write(f"Tingkat Keyakinan Model: **{prediction_proba[0][1] * 100:.2f}%**")
    else:
        st.success("✅ **NORMAL** (Risiko Rendah Penyakit Kardiovaskular).")
        st.write(f"Tingkat Keyakinan Model: **{prediction_proba[0][0] * 100:.2f}%**")