import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import json
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ===============================================================
# KONFIGURASI HALAMAN
# ===============================================================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# LOAD MODEL & ARTEFAK
# ===============================================================
@st.cache_resource
def load_artifacts():
    svm_model = joblib.load("model_svm_optimized.joblib")
    rf_model  = joblib.load("model_rf_optimized.joblib")
    with open("optimal_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return svm_model, rf_model, thresholds

try:
    svm_model, rf_model, optimal_thresholds = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)

# ===============================================================
# SIDEBAR — Input Data Pasien
# ===============================================================
with st.sidebar:
    st.markdown("## 🏥 Masukkan Data Pasien")

    st.markdown("**Pilih Model Prediksi:**")
    algo_choice = st.radio(
        "Model",
        [
            "Tuned Random Forest (Rekomendasi)",
            "Tuned SVM"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Input fields
    age = st.slider("Umur (Age)", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Jenis Kelamin (Sex)", ["M", "F"])
    chest_pain = st.selectbox(
        "Tipe Nyeri Dada (ChestPainType)",
        ["ATA", "NAP", "ASY", "TA"]
    )
    resting_bp = st.slider("Tekanan Darah (RestingBP)", min_value=80, max_value=220, value=120)
    cholesterol = st.slider("Kolesterol (Cholesterol)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox(
        "Gula Darah Puasa > 120? (FastingBS)",
        ["0", "1"]
    )
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )
    max_hr = st.slider("Detak Jantung Maks (MaxHR)", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )
    oldpeak = st.slider("Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi", use_container_width=True)

# ===============================================================
# MAIN AREA
# ===============================================================
st.markdown("## 💗 Sistem Prediksi Penyakit Kardiovaskular")
st.markdown(
    "Aplikasi ini menggunakan model Machine Learning **(SVM & Random Forest)** "
    "untuk memprediksi risiko penyakit jantung berdasarkan data rekam medis pasien."
)

# ── Metrik Evaluasi Model ───────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Hasil Evaluasi Model")

col_svm, col_rf = st.columns(2)

with col_svm:
    st.markdown("### 🔵 Tuned SVM")
    st.markdown(
        "Parameter Terbaik: **kernel**: linear, **gamma**: 0.001, **C**: 50"
    )
    st.metric("Accuracy",  "0.8696")
    st.metric("Precision", "0.8750")
    st.metric("Recall",    "0.8922")
    st.metric("F1-Score",  "0.8835")

with col_rf:
    st.markdown("### 🟢 Tuned Random Forest")
    st.markdown(
        "Parameter Terbaik: **n_estimators**: 300, **min_samples_split**: 5, "
        "**min_samples_leaf**: 2, **max_depth**: None"
    )
    st.metric("Accuracy",  "0.8804")
    st.metric("Precision", "0.8704")
    st.metric("Recall",    "0.9216")
    st.metric("F1-Score",  "0.8953")

# ── Stacking Ensemble ───────────────────────────────────────────


# ── Visualisasi dari PNG ────────────────────────────────────────
st.markdown("---")
st.markdown("## 🖼️ Visualisasi Evaluasi")

tab1, tab2, tab3, tab4 = st.tabs([
    "Confusion Matrix Baseline",
    "Confusion Matrix Optimized",
    "ROC Curve",
    "Perbandingan Metrik"
])

def show_image(tab, path, caption):
    with tab:
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"File `{path}` belum ditemukan. Jalankan training terlebih dahulu.")

show_image(tab1, "baseline_confusion_matrix.png", "Confusion Matrix — Baseline SVM vs Baseline RF")
show_image(tab2, "optimized_evaluation.png",      "Evaluasi Optimized — SVM vs RF vs Stacking")
show_image(tab3, "roc_curve_all.png",             "ROC Curve — Semua Model")
show_image(tab4, "comparison_metrics.png",        "Perbandingan Metrik — Baseline vs Tuned")

# ── Threshold & Parameter Info ──────────────────────────────────
if model_loaded:
    st.markdown("---")
    st.markdown("## ⚙️ Threshold & Parameter Optimal")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Threshold Optimal**")
        st.write(f"- SVM Threshold : `{optimal_thresholds.get('svm_threshold', 0.5):.2f}`")
        st.write(f"- RF Threshold  : `{optimal_thresholds.get('rf_threshold', 0.5):.2f}`")
    with col_t2:
        st.markdown("**Parameter Model**")
        st.write("- SVM Best Kernel : `linear`")
        st.write("- SVM Best C      : `50`")
        st.write("- RF n_estimators : `300`")
        st.write("- RF max_depth    : `None`")

# ── Informasi Dataset & Teknis ──────────────────────────────────
st.markdown("---")
st.markdown("## ℹ️ Tentang Aplikasi")

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("**📁 Informasi Dataset & Model**")
    st.write("- Dataset        : Heart Failure Prediction")
    st.write("- Sumber         : Kaggle (fedesoriano)")
    st.write("- Jumlah data    : 918 pasien")
    st.write("- Fitur          : 11 fitur klinis")
    st.write("- Target         : HeartDisease (0 / 1)")
    st.write("- Split          : 80% train / 20% test")

with col_i2:
    st.markdown("**🤖 Informasi Teknis**")
    st.write("- Algoritma      : SVM & Random Forest")
    st.write("- Tuning         : RandomizedSearchCV")
    st.write("- Cross-val      : 10-Fold CV")
    st.write("- Scoring tuning : Recall")
    st.write("- Framework ML   : Scikit-learn")
    st.write("- UI             : Streamlit")

# ===============================================================
# HASIL PREDIKSI — muncul setelah tombol ditekan
# ===============================================================
if predict_btn:
    if not model_loaded:
        st.error(f"Model gagal dimuat: {load_error}")
    else:
        st.markdown("---")
        st.markdown("## 🔮 Hasil Prediksi")

        # Mapping pilihan model
        if "SVM" in algo_choice:
            selected_model     = svm_model
            selected_label     = "Tuned SVM"
            selected_threshold = optimal_thresholds.get('svm_threshold', 0.5)
        else:
            selected_model     = rf_model
            selected_label     = "Tuned Random Forest"
            selected_threshold = optimal_thresholds.get('rf_threshold', 0.5)

        # Siapkan input DataFrame
        input_dict = {
            "Age"            : age,
            "Sex"            : sex,
            "ChestPainType"  : chest_pain,
            "RestingBP"      : resting_bp,
            "Cholesterol"    : cholesterol,
            "FastingBS"      : int(fasting_bs),
            "RestingECG"     : resting_ecg,
            "MaxHR"          : max_hr,
            "ExerciseAngina" : exercise_angina,
            "Oldpeak"        : oldpeak,
            "ST_Slope"       : st_slope,
        }
        input_df = pd.DataFrame([input_dict])

        # Feature Engineering
        input_df['Age_MaxHR_ratio']   = input_df['Age'] / (input_df['MaxHR'] + 1)
        input_df['Chol_Age_product']  = input_df['Cholesterol'] * input_df['Age'] / 1000
        input_df['BP_age_ratio']      = input_df['RestingBP'] / input_df['Age']
        input_df['MaxHR_Age_diff']    = (220 - input_df['Age']) - input_df['MaxHR']
        input_df['Oldpeak_sq']        = input_df['Oldpeak'] ** 2
        input_df['is_elderly']        = (input_df['Age'] >= 60).astype(int)
        input_df['high_chol']         = (input_df['Cholesterol'] >= 200).astype(int)
        input_df['exercise_capacity'] = input_df['MaxHR'] / (input_df['Age'] + 1)

        # Prediksi
        prob_positive_raw = selected_model.predict_proba(input_df)[0][1]
        prob_positive     = prob_positive_raw * 100
        pred              = int(prob_positive_raw >= selected_threshold)

        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            if pred == 1:
                st.error(f"⚠️ **Risiko Penyakit Jantung Terdeteksi**")
                st.write(f"Model: **{selected_label}** | Threshold: `{selected_threshold:.2f}`")
                st.progress(int(prob_positive))
                st.write(f"Probabilitas risiko: **{prob_positive:.1f}%**")
            else:
                st.success(f"✅ **Tidak Terdeteksi Risiko Penyakit Jantung**")
                st.write(f"Model: **{selected_label}** | Threshold: `{selected_threshold:.2f}`")
                st.progress(int(100 - prob_positive))
                st.write(f"Probabilitas normal: **{100 - prob_positive:.1f}%**")

        with col_r2:
            st.markdown("**Data Pasien yang Diinput:**")
            display_data = {
                "Umur"          : age,
                "Jenis Kelamin" : sex,
                "Nyeri Dada"    : chest_pain,
                "Tekanan Darah" : f"{resting_bp} mmHg",
                "Kolesterol"    : f"{cholesterol} mg/dL",
                "Gula Darah"    : f"{'Ya' if fasting_bs == '1' else 'Tidak'}",
                "Resting ECG"   : resting_ecg,
                "Max HR"        : max_hr,
                "Ex. Angina"    : exercise_angina,
                "Oldpeak"       : oldpeak,
                "ST Slope"      : st_slope,
            }
            st.dataframe(pd.DataFrame(display_data.items(), columns=["Parameter", "Nilai"]),
                         hide_index=True, use_container_width=True)

        st.caption("⚠️ Disclaimer: Hasil prediksi ini hanya bersifat informatif dan tidak menggantikan diagnosis medis profesional.")
