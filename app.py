import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================================
# 1. Konfigurasi Halaman Streamlit
# ============================================================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Sistem Prediksi Penyakit Kardiovaskular")
st.write(
    "Aplikasi ini menggunakan model Machine Learning (SVM & Random Forest) "
    "untuk memprediksi risiko penyakit jantung berdasarkan data rekam medis pasien."
)

# ============================================================
# 2. Load & Cleansing Data (dari notebook)
# ============================================================

@st.cache_data
def load_and_clean_data():
    """Memuat dan membersihkan dataset heart.csv."""
    # Load Data
    df = pd.read_csv('heart.csv')

    # ── Cleansing Data ──────────────────────────────────────
    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    # Nilai 0 tidak valid → ganti dengan median
    for col in ['Cholesterol', 'RestingBP']:
        median = df.loc[df[col] != 0, col].median()
        df[col] = df[col].replace(0, median)

    # Winsorize outlier (1%–99%)
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for col in num_cols:
        df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    return df


# ============================================================
# 3. Bangun & Latih Pipeline (dari notebook)
# ============================================================

@st.cache_resource
def build_and_train_pipelines(df):
    """
    Membangun pipeline preprocessing + model SVM dan Random Forest,
    lalu melatihnya dengan data dari df.
    """
    # ── Split Fitur & Target ────────────────────────────────
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43, stratify=y
    )

    # ── Konfigurasi Preprocessor ────────────────────────────
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

    # ── Pipeline SVM ────────────────────────────────────────
    pipeline_svm = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])

    # ── Pipeline Random Forest ───────────────────────────────
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # ── Latih Pipeline ──────────────────────────────────────
    pipeline_svm.fit(X_train, y_train)
    pipeline_rf.fit(X_train, y_train)

    return pipeline_svm, pipeline_rf, X_train, X_test, y_train, y_test


# ============================================================
# 4. Hyperparameter Tuning dengan RandomizedSearchCV (dari notebook)
# ============================================================

@st.cache_resource
def tune_models(_pipeline_svm, _pipeline_rf, X_train, y_train):
    """
    Melakukan hyperparameter tuning menggunakan RandomizedSearchCV
    untuk SVM dan Random Forest.
    """
    # Ruang Pencarian Parameter SVM
    param_grid_svm = {
        'classifier__C':      [0.1, 1, 10, 50, 100],
        'classifier__gamma':  ['scale', 'auto', 0.1, 0.01, 0.001],
        'classifier__kernel': ['rbf', 'linear']
    }

    # Ruang Pencarian Parameter Random Forest
    param_grid_rf = {
        'classifier__n_estimators':    [50, 100, 200, 300],
        'classifier__max_depth':       [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf':  [1, 2, 4]
    }

    # RandomizedSearchCV – SVM (scoring: roc_auc)
    rs_svm = RandomizedSearchCV(
        estimator=_pipeline_svm,
        param_distributions=param_grid_svm,
        n_iter=15,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # RandomizedSearchCV – Random Forest (scoring: recall)
    rs_rf = RandomizedSearchCV(
        estimator=_pipeline_rf,
        param_distributions=param_grid_rf,
        n_iter=15,
        scoring='recall',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # Eksekusi Tuning
    rs_svm.fit(X_train, y_train)
    rs_rf.fit(X_train, y_train)

    best_svm_pipeline = rs_svm.best_estimator_
    best_rf_pipeline  = rs_rf.best_estimator_

    return best_svm_pipeline, best_rf_pipeline, rs_svm, rs_rf


# ============================================================
# 5. Fungsi Evaluasi (dari notebook)
# ============================================================

def evaluate_baseline(model, X_te, y_te, label, params):
    """Mengevaluasi model dan mengembalikan metrik performa."""
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    accuracy  = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall    = recall_score(y_te, y_pred)
    f1        = f1_score(y_te, y_pred)
    auc       = roc_auc_score(y_te, y_prob)

    return y_pred, y_prob, accuracy, precision, recall, f1, auc


# ============================================================
# 6. Visualisasi (dari notebook)
# ============================================================

def plot_confusion_matrix(y_test, y_pred_svm, y_pred_rf, acc_svm, auc_svm, acc_rf, auc_rf):
    """Menampilkan Confusion Matrix berdampingan untuk SVM dan RF."""
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('Confusion Matrix — Tuned SVM vs Random Forest', fontsize=13, fontweight='bold')

    sns.heatmap(
        confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d',
        cmap='Blues', ax=axes1[0],
        xticklabels=['Normal', 'Heart Disease'],
        yticklabels=['Normal', 'Heart Disease']
    )
    axes1[0].set(
        title=f'Tuned SVM\nAcc={acc_svm:.3f} | AUC={auc_svm:.3f}',
        ylabel='Actual', xlabel='Predicted'
    )

    sns.heatmap(
        confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
        cmap='Greens', ax=axes1[1],
        xticklabels=['Normal', 'Heart Disease'],
        yticklabels=['Normal', 'Heart Disease']
    )
    axes1[1].set(
        title=f'Tuned Random Forest\nAcc={acc_rf:.3f} | AUC={auc_rf:.3f}',
        ylabel='Actual', xlabel='Predicted'
    )

    plt.tight_layout()
    return fig1


def plot_roc_curve(y_test, y_prob_svm, y_prob_rf, auc_svm, auc_rf):
    """Menampilkan ROC Curve gabungan SVM dan RF."""
    fig2, ax = plt.subplots(figsize=(8, 6))
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)

    ax.plot(fpr_svm, tpr_svm, color='steelblue', lw=2, label=f'Tuned SVM (AUC={auc_svm:.3f})')
    ax.plot(fpr_rf,  tpr_rf,  color='seagreen',  lw=2, label=f'Tuned RF  (AUC={auc_rf:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.fill_between(fpr_svm, tpr_svm, alpha=0.05, color='steelblue')
    ax.fill_between(fpr_rf,  tpr_rf,  alpha=0.05, color='seagreen')
    ax.set_title('ROC Curve — Tuned SVM vs Random Forest')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    return fig2


def plot_metric_comparison(acc_svm, pre_svm, rec_svm, f1_svm, auc_svm,
                           acc_rf,  pre_rf,  rec_rf,  f1_rf,  auc_rf):
    """Menampilkan Bar Chart perbandingan metrik evaluasi."""
    fig3, ax = plt.subplots(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    svm_scores   = [acc_svm, pre_svm, rec_svm, f1_svm, auc_svm]
    rf_scores    = [acc_rf,  pre_rf,  rec_rf,  f1_rf,  auc_rf]
    x            = np.arange(len(metric_names))
    width        = 0.35

    bars1 = ax.bar(x - width / 2, svm_scores, width, label='Tuned SVM',
                   color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width / 2, rf_scores,  width, label='Tuned RF',
                   color='seagreen',  alpha=0.85)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title('Perbandingan Metrik — Tuned SVM vs Random Forest')
    ax.set_ylabel('Score')
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig3


# ============================================================
# 7. Jalankan Pipeline di Streamlit
# ============================================================

df = load_and_clean_data()

with st.spinner("⏳ Melatih model baseline (SVM & Random Forest)..."):
    pipeline_svm, pipeline_rf, X_train, X_test, y_train, y_test = build_and_train_pipelines(df)

with st.spinner("⏳ Melakukan hyperparameter tuning... (mungkin butuh beberapa menit pertama kali)"):
    best_svm_pipeline, best_rf_pipeline, rs_svm, rs_rf = tune_models(
        pipeline_svm, pipeline_rf, X_train, y_train
    )

# Evaluasi
y_pred_svm, y_prob_svm, acc_svm, pre_svm, rec_svm, f1_svm, auc_svm = evaluate_baseline(
    best_svm_pipeline, X_test, y_test,
    label="Tuned SVM (Best Model)",
    params=rs_svm.best_params_
)

y_pred_rf, y_prob_rf, acc_rf, pre_rf, rec_rf, f1_rf, auc_rf = evaluate_baseline(
    best_rf_pipeline, X_test, y_test,
    label="Tuned Random Forest (Best Model)",
    params=rs_rf.best_params_
)

# ============================================================
# 8. Tampilan Hasil Evaluasi di Streamlit
# ============================================================

st.header("📊 Hasil Evaluasi Model")

# Tabel ringkasan metrik
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Tuned SVM")
    
    # Format parameter agar lebih rapi (hilangkan 'classifier__' dan kurung kurawal)
    formatted_svm_params = ", ".join([f"**{k.replace('classifier__', '')}**: {v}" for k, v in rs_svm.best_params_.items()])
    st.write(f"Parameter Terbaik: {formatted_svm_params}")
    
    st.metric("Accuracy",  f"{acc_svm:.4f}")
    st.metric("Precision", f"{pre_svm:.4f}")
    st.metric("Recall",    f"{rec_svm:.4f}")
    st.metric("F1-Score",  f"{f1_svm:.4f}")
    st.metric("ROC-AUC",   f"{auc_svm:.4f}")

with col2:
    st.subheader("🟢 Tuned Random Forest")
    
    formatted_rf_params = ", ".join([f"**{k.replace('classifier__', '')}**: {v}" for k, v in rs_rf.best_params_.items()])
    st.write(f"Parameter Terbaik: {formatted_rf_params}")
    
    st.metric("Accuracy",  f"{acc_rf:.4f}")
    st.metric("Precision", f"{pre_rf:.4f}")
    st.metric("Recall",    f"{rec_rf:.4f}")
    st.metric("F1-Score",  f"{f1_rf:.4f}")
    st.metric("ROC-AUC",   f"{auc_rf:.4f}")

# Visualisasi
st.header("📈 Visualisasi Evaluasi")

tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Perbandingan Metrik"])

with tab1:
    fig1 = plot_confusion_matrix(
        y_test, y_pred_svm, y_pred_rf, acc_svm, auc_svm, acc_rf, auc_rf
    )
    st.pyplot(fig1)

with tab2:
    fig2 = plot_roc_curve(y_test, y_prob_svm, y_prob_rf, auc_svm, auc_rf)
    st.pyplot(fig2)

with tab3:
    fig3 = plot_metric_comparison(
        acc_svm, pre_svm, rec_svm, f1_svm, auc_svm,
        acc_rf,  pre_rf,  rec_rf,  f1_rf,  auc_rf
    )
    st.pyplot(fig3)

# ============================================================
# 9. Prediksi Individual Pasien
# ============================================================

st.header("🩺 Prediksi Risiko Pasien Baru")

with st.sidebar:
    st.header("📝 Masukkan Data Pasien")
    model_choice = st.radio(
        "Pilih Model Prediksi:",
        ("Tuned Random Forest (Rekomendasi)", "Tuned SVM")
    )

    age            = st.slider("Umur (Age)", 20, 100, 50)
    sex            = st.selectbox("Jenis Kelamin (Sex)", ("M", "F"))
    chest_pain     = st.selectbox("Tipe Nyeri Dada (ChestPainType)", ("ATA", "NAP", "ASY", "TA"))
    resting_bp     = st.slider("Tekanan Darah (RestingBP)", 80, 200, 120)
    cholesterol    = st.slider("Kolesterol (Cholesterol)", 100, 400, 200)
    fasting_bs     = st.selectbox("Gula Darah Puasa > 120? (FastingBS)", (0, 1))
    resting_ecg    = st.selectbox("Hasil EKG (RestingECG)", ("Normal", "ST", "LVH"))
    max_hr         = st.slider("Detak Jantung Maksimal (MaxHR)", 60, 220, 150)
    exercise_angina = st.selectbox("Nyeri Dada saat Olahraga? (ExerciseAngina)", ("Y", "N"))
    oldpeak        = st.slider("Depresi ST (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    st_slope       = st.selectbox("Kemiringan ST (ST_Slope)", ("Up", "Flat", "Down"))


def build_input_df():
    data = {
        'Age':            age,
        'Sex':            sex,
        'ChestPainType':  chest_pain,
        'RestingBP':      resting_bp,
        'Cholesterol':    cholesterol,
        'FastingBS':      fasting_bs,
        'RestingECG':     resting_ecg,
        'MaxHR':          max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak':        oldpeak,
        'ST_Slope':       st_slope
    }
    return pd.DataFrame(data, index=[0])


# ============================================================
# 10. Fungsi Nasihat Kesehatan Berdasarkan Data Pasien
# ============================================================

def generate_health_advice(age, sex, chest_pain, resting_bp, cholesterol,fasting_bs, resting_ecg, max_hr,exercise_angina, oldpeak, st_slope):
    """
    Menganalisis data klinis pasien dan menghasilkan daftar nasihat
    kesehatan yang relevan berdasarkan nilai yang di luar batas normal.
    """
    advice_list = []

    # --- Usia ---
    if age >= 60:
        advice_list.append({
            "icon": "🧓",
            "judul": "Faktor Usia",
            "isi": (
                "Risiko penyakit jantung meningkat seiring bertambahnya usia. "
                "Lakukan pemeriksaan jantung rutin setidaknya sekali setahun bersama dokter spesialis "
                "jantung (kardiolog) dan ikuti program gaya hidup sehat yang direkomendasikan dokter."
            )
        })

    # --- Tekanan Darah (RestingBP) ---
    if resting_bp >= 130:
        advice_list.append({
            "icon": "🩺",
            "judul": "Tekanan Darah Tinggi (Hipertensi)",
            "isi": (
                f"Tekanan darah istirahat Anda **{resting_bp} mmHg** berada di atas batas normal (< 120 mmHg). "
                "Kurangi konsumsi garam dan makanan olahan, perbanyak konsumsi buah dan sayur, "
                "olahraga ringan seperti jalan kaki 30 menit per hari, kelola stres dengan baik, "
                "dan konsultasikan ke dokter untuk kemungkinan pemberian obat antihipertensi."
            )
        })

    # --- Kolesterol ---
    if cholesterol >= 200:
        advice_list.append({
            "icon": "🧪",
            "judul": "Kadar Kolesterol Tinggi",
            "isi": (
                f"Kadar kolesterol Anda **{cholesterol} mg/dL** di atas batas ideal (< 200 mg/dL). "
                "Hindari makanan berlemak jenuh dan trans (gorengan, daging berlemak), "
                "perbanyak konsumsi ikan, kacang-kacangan, oatmeal, dan buah-buahan. "
                "Pertimbangkan terapi obat statin jika direkomendasikan dokter."
            )
        })

    # --- Detak Jantung Maksimal (MaxHR) ---
    max_hr_normal = 220 - age
    if max_hr > max_hr_normal * 0.95:
        advice_list.append({
            "icon": "💓",
            "judul": "Detak Jantung Saat Olahraga Sangat Tinggi",
            "isi": (
                f"Detak jantung maksimal Anda **{max_hr} bpm** mendekati atau melebihi batas aman "
                f"(~{int(max_hr_normal * 0.85)}–{max_hr_normal} bpm untuk usia {age} tahun). "
                "Hindari olahraga berlebihan tanpa pengawasan, gunakan monitor detak jantung saat berolahraga, "
                "dan mulai dengan latihan intensitas rendah-sedang seperti jalan santai atau bersepeda santai."
            )
        })
    elif max_hr < 100:
        advice_list.append({
            "icon": "💔",
            "judul": "Detak Jantung Maksimal Rendah (Bradikardia)",
            "isi": (
                f"Detak jantung maksimal Anda **{max_hr} bpm** tergolong rendah saat aktivitas. "
                "Ini bisa mengindikasikan kondisi bradikardia atau keterbatasan fungsi jantung. "
                "Segera konsultasikan ke dokter untuk evaluasi lebih lanjut seperti EKG atau tes stres jantung."
            )
        })

    # --- Gula Darah Puasa (FastingBS) ---
    if fasting_bs == 1:
        advice_list.append({
            "icon": "🍬",
            "judul": "Gula Darah Puasa Tinggi (> 120 mg/dL)",
            "isi": (
                "Kadar gula darah puasa Anda **di atas 120 mg/dL**, yang merupakan indikator risiko diabetes. "
                "Kurangi konsumsi makanan dan minuman manis, nasi putih, dan karbohidrat olahan. "
                "Tingkatkan aktivitas fisik, pantau gula darah secara rutin, dan konsultasikan ke dokter "
                "untuk tes HbA1c guna mengetahui kondisi diabetes lebih lanjut."
            )
        })

    # --- Nyeri Dada saat Olahraga (ExerciseAngina) ---
    if exercise_angina == "Y":
        advice_list.append({
            "icon": "⚠️",
            "judul": "Nyeri Dada saat Beraktivitas (Angina)",
            "isi": (
                "Anda mengalami **nyeri dada saat berolahraga**, yang merupakan tanda serius penyempitan "
                "pembuluh darah koroner. **Segera hentikan** aktivitas fisik berat dan konsultasikan ke "
                "dokter jantung sesegera mungkin. Jangan tunda pemeriksaan karena ini berisiko tinggi "
                "menyebabkan serangan jantung."
            )
        })

    # --- Depresi ST (Oldpeak) ---
    if oldpeak > 2.0:
        advice_list.append({
            "icon": "📉",
            "judul": "Depresi Segmen ST Tinggi",
            "isi": (
                f"Nilai depresi ST (Oldpeak) Anda **{oldpeak}**, jauh di atas ambang normal (≤ 1.0). "
                "Ini menandakan adanya iskemia (kekurangan aliran darah ke otot jantung) yang signifikan. "
                "Diperlukan pemeriksaan lanjutan seperti kateterisasi jantung atau angiografi koroner. "
                "Konsultasikan segera ke dokter spesialis jantung."
            )
        })
    elif oldpeak > 1.0:
        advice_list.append({
            "icon": "📊",
            "judul": "Depresi Segmen ST Sedang",
            "isi": (
                f"Nilai depresi ST (Oldpeak) Anda **{oldpeak}** sedikit di atas normal. "
                "Lakukan pemantauan berkala dengan EKG dan konsultasikan hasilnya ke dokter."
            )
        })

    # --- Kemiringan ST (ST_Slope) ---
    if st_slope == "Down":
        advice_list.append({
            "icon": "📐",
            "judul": "Kemiringan ST Menurun (Downsloping)",
            "isi": (
                "Pola **ST slope menurun** pada EKG Anda adalah indikator kuat adanya masalah pada "
                "aliran darah koroner. Ini memerlukan evaluasi kardiologi segera, termasuk "
                "kemungkinan dilakukannya tes treadmill atau stress echo cardiography."
            )
        })
    elif st_slope == "Flat":
        advice_list.append({
            "icon": "📏",
            "judul": "Kemiringan ST Datar (Flat)",
            "isi": (
                "Pola **ST slope datar** menunjukkan respons jantung yang kurang optimal saat beraktivitas. "
                "Lakukan monitoring EKG secara berkala dan diskusikan dengan dokter jantung Anda."
            )
        })

    # --- Tipe Nyeri Dada ---
    if chest_pain == "ASY":
        advice_list.append({
            "icon": "🫀",
            "judul": "Nyeri Dada Asimptomatik (Tanpa Gejala Nyata)",
            "isi": (
                "Penyakit jantung tanpa gejala yang jelas (asimptomatik) justru sangat berbahaya karena "
                "sering terlambat dideteksi. Lakukan **skrining jantung rutin** meski tidak ada keluhan, "
                "termasuk pemeriksaan EKG, ekokardiografi, dan profil lipid darah setidaknya setahun sekali."
            )
        })

    # --- Hasil EKG ---
    if resting_ecg == "ST":
        advice_list.append({
            "icon": "🏥",
            "judul": "Kelainan Segmen ST pada EKG",
            "isi": (
                "Hasil EKG Anda menunjukkan **kelainan segmen ST** yang perlu dievaluasi lebih lanjut. "
                "Segera hubungi dokter untuk interpretasi hasil EKG dan pertimbangkan pemeriksaan lanjutan "
                "seperti Holter monitoring atau ekokardiografi."
            )
        })

    # Jika tidak ada faktor spesifik yang terdeteksi (nasihat umum)
    if not advice_list:
        advice_list.append({
            "icon": "💊",
            "judul": "Saran Umum Pola Hidup Sehat",
            "isi": (
                "Meskipun tidak ada faktor risiko tunggal yang sangat menonjol, model memprediksi risiko tinggi "
                "berdasarkan kombinasi faktor. Disarankan untuk: berhenti merokok (jika perokok), "
                "menjaga berat badan ideal, rutin berolahraga 150 menit per minggu, makan makanan bergizi "
                "seimbang, dan melakukan pemeriksaan jantung menyeluruh bersama dokter."
            )
        })

    return advice_list


input_df = build_input_df()

st.subheader("Data Pasien Saat Ini")
st.write(input_df)

if st.button("🔍 Lakukan Prediksi", type="primary"):
    # Pilih model sesuai pilihan user
    chosen_model = best_rf_pipeline if "Random Forest" in model_choice else best_svm_pipeline

    prediction       = chosen_model.predict(input_df)
    prediction_proba = chosen_model.predict_proba(input_df)

    st.subheader("Hasil Prediksi")

    if prediction[0] == 1:
        st.error("⚠️ **BERISIKO TINGGI** mengidap Penyakit Kardiovaskular.")
        st.write(f"Tingkat Keyakinan Model: **{prediction_proba[0][1] * 100:.2f}%**")

        # ── Nasihat Kesehatan Berdasarkan Data Pasien ────────────
        advice_items = generate_health_advice(
            age, sex, chest_pain, resting_bp, cholesterol,
            fasting_bs, resting_ecg, max_hr,
            exercise_angina, oldpeak, st_slope
        )

        st.markdown("---")
        st.subheader("💡 Nasihat & Rekomendasi Kesehatan")
        st.caption(
            "Berikut adalah rekomendasi berdasarkan kondisi klinis yang terdeteksi dari data pasien. "
            "**Konsultasikan selalu dengan dokter atau tenaga medis profesional.**"
        )

        for item in advice_items:
            with st.expander(f"{item['icon']}  {item['judul']}", expanded=True):
                st.markdown(item['isi'])

        st.info(
            "🏥 **Penting:** Hasil prediksi ini bersifat sebagai alat bantu skrining awal, "
            "bukan diagnosis medis resmi. Segera kunjungi dokter spesialis jantung untuk "
            "pemeriksaan dan penanganan yang tepat."
        )

    else:
        st.success("✅ **NORMAL** (Risiko Rendah Penyakit Kardiovaskular).")
        st.write(f"Tingkat Keyakinan Model: **{prediction_proba[0][0] * 100:.2f}%**")