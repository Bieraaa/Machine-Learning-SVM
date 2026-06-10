import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ===============================================================
# KONFIGURASI HALAMAN
# ===============================================================
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# CUSTOM CSS
# ===============================================================
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #f8f8f8; border-right: 1px solid #ececec; }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        display: flex; align-items: center; gap: 10px;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 0.5rem;
    }
    .brand-icon { font-size: 26px; }
    .brand-text { font-size: 15px; font-weight: 600; color: #2c2c2c; line-height: 1.2; }
    .brand-sub  { font-size: 11px; color: #999; font-weight: 400; }

    /* ── Page header ── */
    .page-header { margin-bottom: 1.5rem; }
    .page-title  { font-size: 22px; font-weight: 600; color: #1a1a1a; margin: 0; }
    .page-sub    { font-size: 13px; color: #888; margin-top: 3px; }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 13px; font-weight: 600;
        color: #666; margin-bottom: 14px;
        text-transform: uppercase; letter-spacing: 0.04em;
        display: flex; align-items: center; gap: 6px;
    }

    /* ── Metric cards ── */
    .metrics-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem; }
    .metric-card {
        flex: 1; min-width: 100px;
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val { font-size: 22px; font-weight: 700; color: #1a1a1a; }
    .metric-lbl { font-size: 11px; color: #999; margin-top: 3px; }

    /* ── Result box ── */
    .result-danger {
        background: #fff5f5;
        border: 1.5px solid #f5c6c6;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-safe {
        background: #f5fff8;
        border: 1.5px solid #b7e5c4;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-title-danger { font-size: 18px; font-weight: 700; color: #8b1a1a; margin: 0; }
    .result-title-safe   { font-size: 18px; font-weight: 700; color: #1a5c2a; margin: 0; }
    .result-sub-danger   { font-size: 13px; color: #b94040; margin-top: 4px; }
    .result-sub-safe     { font-size: 13px; color: #2e7d45; margin-top: 4px; }

    /* ── Prob bar ── */
    .prob-wrap { margin-top: 14px; }
    .prob-label { display: flex; justify-content: space-between; font-size: 12px; color: #888; margin-bottom: 5px; }
    .prob-bg { background: #f0f0f0; border-radius: 99px; height: 8px; overflow: hidden; }
    .prob-fill-danger { height: 100%; border-radius: 99px; background: #c0392b; }
    .prob-fill-safe   { height: 100%; border-radius: 99px; background: #27ae60; }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fffbf0;
        border: 1px solid #f5e6b0;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        font-size: 12px;
        color: #7a6000;
        line-height: 1.6;
    }

    /* ── Info row ── */
    .info-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 7px 0; border-bottom: 1px solid #f0f0f0;
        font-size: 13px;
    }
    .info-row:last-child { border-bottom: none; }
    .info-key { color: #888; }
    .info-val { font-weight: 600; color: #1a1a1a; }

    /* ── Badge ── */
    .badge-maroon {
        display: inline-block;
        background: #6b1a1a; color: #fff;
        font-size: 11px; font-weight: 600;
        padding: 3px 10px; border-radius: 6px;
        vertical-align: middle; margin-left: 6px;
    }
    .badge-gray {
        display: inline-block;
        background: #ececec; color: #555;
        font-size: 11px; font-weight: 600;
        padding: 3px 10px; border-radius: 6px;
        vertical-align: middle; margin-left: 6px;
    }

    /* ── Form ── */
    .stSelectbox label, .stNumberInput label { font-size: 13px !important; color: #555 !important; }
    div[data-testid="stNumberInput"] input { border-radius: 8px !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #6b1a1a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0.55rem 1.5rem !important;
        width: 100% !important;
        transition: background 0.15s !important;
    }
    .stButton > button:hover { background: #8b2020 !important; }

    /* ── Sidebar nav radio ── */
    div[data-testid="stRadio"] label {
        font-size: 14px !important;
        color: #444 !important;
    }
    div[data-testid="stRadio"] > div { gap: 2px !important; }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #ececec; margin: 1rem 0; }

    /* hide streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; }

    /* ── Sidebar toggle button — multi-selector untuk semua versi Streamlit ── */

    /* Tombol panah ketika sidebar TERTUTUP (collapsed) */
    [data-testid="collapsedControl"] {
        background-color: #6b1a1a !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 6px 4px !important;
        box-shadow: 3px 0 10px rgba(107,26,26,0.5) !important;
        opacity: 1 !important;
        min-width: 28px !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="collapsedControl"] button {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    /* Tombol panah ketika sidebar TERBUKA (expand/collapse di dalam sidebar) */
    section[data-testid="stSidebar"] button {
        background-color: #6b1a1a !important;
        border-radius: 50% !important;
        color: #ffffff !important;
        opacity: 1 !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] button svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background-color: #8b2020 !important;
        transform: scale(1.1) !important;
    }
</style>
""", unsafe_allow_html=True)


# ===============================================================
# LOAD MODEL & ARTEFAK
# ===============================================================
import json

@st.cache_resource
def load_artifacts():
    """Load model hasil CardioVascular_Optimized.py (Stacking Ensemble)."""
    stacking   = joblib.load("model_stacking.joblib")
    svm_model  = joblib.load("model_svm_optimized.joblib")
    rf_model   = joblib.load("model_rf_optimized.joblib")
    with open("optimal_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return stacking, svm_model, rf_model, thresholds

try:
    stacking_model, svm_model, rf_model, optimal_thresholds = load_artifacts()
    model_loaded = True
    best_label   = "Stacking Ensemble"
except Exception as e:
    model_loaded = False
    load_error   = str(e)

# ===============================================================
# SIDEBAR — hanya info & branding
# ===============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🫀</div>
        <div>
            <div class="brand-text">Heart Disease AI</div>
            <div class="brand-sub">Sistem prediksi berbasis ML</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown(f"""
        <div style="font-size:12px; color:#aaa; margin-bottom:6px;">Model aktif</div>
        <div style="font-size:14px; font-weight:600; color:#1a1a1a;">
            {best_label} <span class="badge-maroon">Optimized</span>
        </div>
        <div style="font-size:12px; color:#aaa; margin-top:10px;">AUC: 0.9454 &nbsp;|&nbsp; Acc: 86.96%</div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model tidak ditemukan. Jalankan CardioVascular_Optimized.py dulu.")

# ===============================================================
# NAVIGASI TABS — selalu tampil di atas halaman
# ===============================================================
tab_prediksi, tab_dashboard, tab_tentang = st.tabs([
    "🩺  Prediksi", "📊  Dashboard Model", "ℹ️  Tentang"
])



# ===============================================================
# HALAMAN 1 — PREDIKSI
# ===============================================================
with tab_prediksi:

    st.markdown("""
    <div class="page-header">
        <div class="page-title">🩺 Prediksi Penyakit Jantung</div>
        <div class="page-sub">Masukkan data klinis pasien untuk mendapatkan hasil prediksi model</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model gagal dimuat: {load_error}")
    else:
        # ── FORM INPUT ──────────────────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">📋 Data Pasien</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            age        = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=52)
            sex        = st.selectbox("Jenis kelamin", ["Laki-laki (M)", "Perempuan (F)"])
            resting_bp = st.number_input("Tekanan darah istirahat (mmHg)", min_value=0, max_value=300, value=140)
            cholesterol = st.number_input("Kolesterol (mg/dL)", min_value=0, max_value=700, value=268)

        with col2:
            chest_pain = st.selectbox("Jenis nyeri dada", ["ASY", "ATA", "NAP", "TA"])
            max_hr     = st.number_input("Detak jantung maksimum", min_value=50, max_value=250, value=125)
            fasting_bs = st.selectbox("Gula darah puasa > 120 mg/dL", ["Tidak (0)", "Ya (1)"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

        with col3:
            exercise_angina = st.selectbox("Exercise Angina", ["Tidak (N)", "Ya (Y)"])
            oldpeak         = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=1.5, step=0.1)
            st_slope        = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

        st.markdown('</div>', unsafe_allow_html=True)

        # ── PILIHAN ALGORITMA ────────────────────────────────────────
        st.markdown('<div class="card"><div class="card-title">🤖 Pilih Algoritma</div>', unsafe_allow_html=True)
        algo_choice = st.radio(
            "Algoritma",
            ["Stacking Ensemble (SVM + RF + LR)", "SVM (Support Vector Machine)", "Random Forest"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── TOMBOL PREDIKSI ─────────────────────────────────────────
        predict_clicked = st.button("🔍 Prediksi Sekarang")

        if predict_clicked:
            sex_val    = "M" if "M" in sex else "F"
            angina_val = "Y" if "Y" in exercise_angina else "N"
            bs_val     = 1 if "1" in fasting_bs else 0

            input_dict = {
                "Age"            : age,
                "Sex"            : sex_val,
                "ChestPainType"  : chest_pain,
                "RestingBP"      : resting_bp,
                "Cholesterol"    : cholesterol,
                "FastingBS"      : bs_val,
                "RestingECG"     : resting_ecg,
                "MaxHR"          : max_hr,
                "ExerciseAngina" : angina_val,
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

            # Pilih model
            if "SVM" in algo_choice and "Stacking" not in algo_choice:
                selected_model     = svm_model
                selected_label     = "Optimized SVM"
                selected_threshold = optimal_thresholds.get('svm_threshold', 0.5)
            elif "Random Forest" in algo_choice:
                selected_model     = rf_model
                selected_label     = "Optimized Random Forest"
                selected_threshold = optimal_thresholds.get('rf_threshold', 0.5)
            else:
                selected_model     = stacking_model
                selected_label     = "Stacking Ensemble"
                selected_threshold = 0.5

            prob_positive_raw = selected_model.predict_proba(input_df)[0][1]
            prob_positive     = prob_positive_raw * 100
            pred              = int(prob_positive_raw >= selected_threshold)

            if pred == 1:
                st.markdown(f"""
                <div class="result-danger">
                    <div style="font-size:13px; color:#b94040; font-weight:600; margin-bottom:4px;">HASIL PREDIKSI</div>
                    <div class="result-title-danger">⚠️ Risiko Penyakit Jantung Terdeteksi</div>
                    <div class="result-sub-danger">Model: {selected_label} &nbsp;·&nbsp; Threshold: {selected_threshold:.2f}</div>
                    <div class="prob-wrap">
                        <div class="prob-label"><span>Probabilitas risiko</span><span>{prob_positive:.1f}%</span></div>
                        <div class="prob-bg"><div class="prob-fill-danger" style="width:{prob_positive:.1f}%;"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <div style="font-size:13px; color:#2e7d45; font-weight:600; margin-bottom:4px;">HASIL PREDIKSI</div>
                    <div class="result-title-safe">✅ Tidak Terdeteksi Risiko Penyakit Jantung</div>
                    <div class="result-sub-safe">Model: {selected_label} &nbsp;·&nbsp; Threshold: {selected_threshold:.2f}</div>
                    <div class="prob-wrap">
                        <div class="prob-label"><span>Probabilitas normal</span><span>{100 - prob_positive:.1f}%</span></div>
                        <div class="prob-bg"><div class="prob-fill-safe" style="width:{100 - prob_positive:.1f}%;"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ===============================================================
# HALAMAN 2 — DASHBOARD MODEL
# ===============================================================
with tab_dashboard:

    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Dashboard Performa Model</div>
        <div class="page-sub">Perbandingan metrik baseline vs tuned — SVM & Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    if model_loaded:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🏆 Model Terbaik (Hasil Optimisasi)</div>
            <div style="font-size:20px; font-weight:700; color:#6b1a1a; margin-bottom:14px;">
                {best_label}
            </div>
            <div class="metrics-row">
                <div class="metric-card"><div class="metric-val" style="color:#6b1a1a;">86.96%</div><div class="metric-lbl">Accuracy</div></div>
                <div class="metric-card"><div class="metric-val" style="color:#6b1a1a;">0.9020</div><div class="metric-lbl">Recall</div></div>
                <div class="metric-card"><div class="metric-val" style="color:#6b1a1a;">0.8679</div><div class="metric-lbl">Precision</div></div>
                <div class="metric-card"><div class="metric-val" style="color:#6b1a1a;">0.8846</div><div class="metric-lbl">F1-Score</div></div>
                <div class="metric-card"><div class="metric-val" style="color:#6b1a1a;">0.9454</div><div class="metric-lbl">ROC-AUC</div></div>
            </div>
            <div style="font-size:12px; color:#aaa;">
                Stacking Ensemble (SVM + Random Forest + Logistic Regression) dengan feature engineering 8 fitur baru.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model belum dimuat. Jalankan CardioVascular_Optimized.py terlebih dahulu.")

    # ── Visualisasi dari PNG yang sudah disimpan ─────────────────
    st.markdown("---")

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

    # ── Info threshold optimal ───────────────────────────────
    if model_loaded:
        st.markdown("<div class='card'><div class='card-title'>⚙️ Threshold Optimal</div>", unsafe_allow_html=True)
        threshold_info = {
            "SVM Threshold"       : f"{optimal_thresholds.get('svm_threshold', 0.5):.2f}",
            "RF Threshold"        : f"{optimal_thresholds.get('rf_threshold', 0.5):.2f}",
            "Stacking Threshold"  : "0.50 (default)",
            "SVM Best Kernel"     : "linear",
            "SVM Best C"          : "1",
            "RF n_estimators"     : "100",
            "RF max_depth"        : "10",
        }
        for k, v in threshold_info.items():
            st.markdown(f"""
            <div class="info-row">
                <span class="info-key">{k}</span>
                <span class="info-val">{v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ===============================================================
# HALAMAN 3 — TENTANG
# ===============================================================
with tab_tentang:

    st.markdown("""
    <div class="page-header">
        <div class="page-title">ℹ️ Tentang Aplikasi</div>
        <div class="page-sub">Informasi project, dataset, dan tim pengembang</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📁 Informasi Dataset & Model</div>
            <div class="info-row"><span class="info-key">Dataset</span><span class="info-val">Heart Failure Prediction</span></div>
            <div class="info-row"><span class="info-key">Sumber</span><span class="info-val">Kaggle (fedesoriano)</span></div>
            <div class="info-row"><span class="info-key">Jumlah data</span><span class="info-val">918 pasien</span></div>
            <div class="info-row"><span class="info-key">Fitur</span><span class="info-val">11 fitur klinis</span></div>
            <div class="info-row"><span class="info-key">Target</span><span class="info-val">HeartDisease (0 / 1)</span></div>
            <div class="info-row"><span class="info-key">Split</span><span class="info-val">80% train / 20% test</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 Informasi Teknis</div>
            <div class="info-row"><span class="info-key">Algoritma</span><span class="info-val">SVM & Random Forest</span></div>
            <div class="info-row"><span class="info-key">Tuning</span><span class="info-val">RandomizedSearchCV</span></div>
            <div class="info-row"><span class="info-key">Cross-validation</span><span class="info-val">10-Fold CV</span></div>
            <div class="info-row"><span class="info-key">Scoring tuning</span><span class="info-val">Recall</span></div>
            <div class="info-row"><span class="info-key">Framework ML</span><span class="info-val">Scikit-learn</span></div>
            <div class="info-row"><span class="info-key">UI</span><span class="info-val">Streamlit</span></div>
        </div>
        """, unsafe_allow_html=True)


