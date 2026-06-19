# =============================================================================
# CardioVascular.py
# =============================================================================
# Script utama untuk melatih dan mengevaluasi model Machine Learning
# dalam memprediksi risiko penyakit jantung (kardiovaskular).
#
# Alur kerja:
#   1. Load Data          → membaca dataset heart.csv
#   2. Pra-Proses Data    → menangani duplikat, missing value, outlier
#   3. Data Transformation → encoding kategorikal (One-Hot Encoding)
#   4. Scaling            → normalisasi fitur numerik (wajib untuk SVM)
#   5. Baseline Model     → melatih SVM & RF dengan parameter default
#   6. Hyperparameter Tuning → optimasi parameter dengan RandomizedSearchCV
#   7. Evaluasi           → menghitung metrik & membandingkan semua model
#   8. Simpan Model       → menyimpan model & artefak ke folder Save_model/
#   9. Visualisasi        → membuat grafik confusion matrix, ROC, & metrik
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, roc_curve,confusion_matrix, classification_report)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# Membaca dataset Heart Failure Prediction dari file CSV
df = pd.read_csv("heart.csv")
print(f"Shape awal: {df.shape}\n")
print(df.head())

# =============================================================================
# 2. PRA-PROSES DATA
# =============================================================================

# Cek missing values — memastikan tidak ada nilai kosong
print(df.isnull().sum())        
print(df.isnull().sum().sum()) 

# Hapus duplikat — menghindari data ganda yang dapat memengaruhi pelatihan
df.drop_duplicates(inplace=True)
print(f"\nShape setelah drop duplikat: {df.shape}")
print(df.head())

# Cek Distribusi data — memastikan proporsi kelas target (HeartDisease)
print("\nDistribusi target (HeartDisease):")
print(df['HeartDisease'].value_counts())
print(df['HeartDisease'].value_counts(normalize=True).round(3))

# Train-Test Split dilakukan SEBELUM preprocessing untuk menghindari data leakage
# stratify=y memastikan proporsi kelas seimbang di train & test
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")

# Nilai 0 pada Cholesterol & RestingBP tidak valid secara medis
# → diganti dengan median dari data training (bukan seluruh data)
for col in ['Cholesterol', 'RestingBP']:
    # Hitung median HANYA dari train (exclude nilai 0)
    median_train = X_train.loc[X_train[col] != 0, col].median()
    X_train[col] = X_train[col].replace(0, median_train)
    X_test[col]  = X_test[col].replace(0, median_train)   # pakai median train
    print(f"Median {col} (dari train): {median_train:.2f}")

# Winsorize outlier pada rentang 1%–99% dari data training
# Teknik ini membatasi nilai ekstrem tanpa menghapus baris data
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
clip_bounds = {}

for col in num_cols:
    low  = X_train[col].quantile(0.01)
    high = X_train[col].quantile(0.99)
    clip_bounds[col] = (low, high)
    X_train[col] = X_train[col].clip(low, high)
    X_test[col]  = X_test[col].clip(low, high) 

print("\nWinsorize bounds (dari train):")
for col, (lo, hi) in clip_bounds.items():
    print(f"  {col}: [{lo:.2f}, {hi:.2f}]")

# =============================================================================
# 3. DATA TRANSFORMATION — One-Hot Encoding
# =============================================================================

# One-Hot Encoding pada fitur kategorikal
# drop='first' menghindari multikolinearitas (dummy variable trap)
# handle_unknown='ignore' menangani kategori baru saat inferensi
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
num_features = [c for c in X_train.columns if c not in cat_cols]

preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
])

# fit hanya dari train
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)   # hanya transform, tidak fit ulang

# Ambil nama fitur hasil encoding (untuk referensi & simpan)
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
feature_names = num_features + list(ohe_feature_names)
print(f"\nJumlah fitur setelah encoding: {len(feature_names)}")
print(f"Fitur: {feature_names}")

# =============================================================================
# 4. SCALING — StandardScaler
# =============================================================================

# StandardScaler wajib untuk SVM karena algoritma ini sensitif terhadap skala fitur
# fit hanya dari data training, transform diterapkan ke keduanya
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_enc)
X_test_sc  = scaler.transform (X_test_enc)

print(f"\nTrain scaled: {X_train_sc.shape} | Test scaled: {X_test_sc.shape}")

# =============================================================================
# 5. BASELINE MODEL
# =============================================================================

# Baseline SVM — menggunakan parameter default sebagai pembanding awal
# SVM menggunakan data yang sudah di-scale (X_train_sc)
svm_base = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_base.fit(X_train_sc, y_train)
print("\nBaseline SVM selesai dilatih.")

# Baseline Random Forest — menggunakan parameter default
# RF tidak memerlukan scaling, menggunakan data encoded (X_train_enc)
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train_enc, y_train)
print("Baseline Random Forest selesai dilatih.")

# =============================================================================
# 6. HYPERPARAMETER TUNING — RandomizedSearchCV
# =============================================================================

# Search space SVM:
# - C      : mengontrol trade-off antara margin dan kesalahan klasifikasi
# - gamma  : menentukan jangkauan pengaruh satu data point
# - kernel : fungsi kernel yang menentukan bentuk batas keputusan
parameter_SVM = {
    'C' : [0.1, 1, 5, 10, 50, 100],
    'gamma' : ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel' : ['rbf', 'linear'],
    'class_weight': ['balanced', None]
}

# Search space Random Forest:
# - n_estimators     : jumlah pohon keputusan
# - max_depth        : kedalaman maksimum setiap pohon
# - min_samples_split: jumlah minimum sampel untuk membagi node
# - min_samples_leaf : jumlah minimum sampel di setiap daun
# - max_features     : jumlah fitur yang dipertimbangkan setiap split
parameter_RF = {
    'n_estimators' : [ 200, 300, 500],
    'max_depth' : [None, 10, 15],
    'min_samples_split' : [2, 5],
    'min_samples_leaf' : [1, 2],
    'max_features' : ['sqrt','log2'],
    'class_weight'   : [ None]
}

#---------------- Randomized Search -----------------------
# scoring='roc_auc' dipilih karena lebih robust untuk dataset dengan kelas tidak seimbang
# cv=5 menggunakan 5-Fold Cross Validation
# n_jobs=-1 memanfaatkan semua core CPU untuk mempercepat proses

search_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    parameter_SVM, n_iter=60, cv= 5,
    scoring='roc_auc', random_state=42, n_jobs=-1, refit=True
)

search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    parameter_RF, n_iter=60, cv=5,
    scoring= 'roc_auc', random_state=42, n_jobs= -1, refit=True
)

#------------------- Tuning ------------------------------

print("Tuning SVM.... ", end='', flush=True)
search_svm.fit(X_train_sc, y_train)
print("Selesai... ")

print("Tuning RF... ", end='', flush=True)
search_rf.fit(X_train_enc, y_train)
print("selesai... ")

print("\n -- Best paramater SVM -- ")
print(search_svm.best_params_)
print("\n -- Best parameter RF -- ")
print(search_rf.best_params_)

# =============================================================================
# 7. EVALUASI MODEL
# =============================================================================

def evaluate_model(model, X_te, y_te, label, params):
    """
    Mengevaluasi performa model klasifikasi dan mencetak hasilnya.

    Parameters
    ----------
    model  : estimator sklearn yang sudah di-fit
    X_te   : array-like, data fitur test
    y_te   : array-like, label target test
    label  : str, nama model untuk ditampilkan
    params : dict, parameter model yang digunakan

    Returns
    -------
    tuple : (y_pred, y_prob, accuracy, precision, recall, f1, auc)
    """
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    accuracy  = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred)
    recall    = recall_score(y_te, y_pred)
    f1        = f1_score(y_te, y_pred)
    auc       = roc_auc_score(y_te, y_prob)

    print("\n" + "=" * 45)
    print(f" {label}")
    print("=" * 45)
    for k, v in params.items():
        print(f"  {k:<12}: {v}")
    print("-" * 45)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("=" * 45)
    print(f"\nClassification Report — {label}:")
    print(classification_report(y_te, y_pred, target_names=['Normal', 'Heart Disease']))

    return y_pred, y_prob, accuracy, precision, recall, f1, auc

# Evaluasi SVM
y_pred_svm_base, y_prob_svm_base, acc_svm_b, pre_svm_b, rec_svm_b, f1_svm_b, auc_svm_b = evaluate_model(
    svm_base, X_test_sc, y_test,
    label="BASELINE SVM",
    params={'Kernel': 'rbf', 'C': '1.0', 'Gamma': 'scale'}
)

# Evaluasi Random Forest
y_pred_rf_base, y_prob_rf_base, acc_rf_b, pre_rf_b, rec_rf_b, f1_rf_b, auc_rf_b = evaluate_model(
    rf_base, X_test_enc, y_test,
    label="BASELINE Random Forest",
    params={'n_estimators': '100', 'max_depth': 'None', 'max_features': 'sqrt'}
)

# Tuned SVM
y_pred_svm_tuned, y_prob_svm_tuned, acc_svm_t, pre_svm_t, rec_svm_t, f1_svm_t, auc_svm_t = evaluate_model(
    search_svm, X_test_sc, y_test,
    label="TUNED SVM",
    params=search_svm.best_params_
)

y_pred_rf_tuned, y_prob_rf_tuned, acc_rf_t, pre_rf_t, rec_rf_t, f1_rf_t, auc_rf_t = evaluate_model(
    search_rf, X_test_enc, y_test,
    label="TUNED Random Forest",
    params=search_rf.best_params_
)

# =============================================================================
# 8. TABEL PERBANDINGAN & PILIH MODEL TERBAIK
# =============================================================================

df_compare = pd.DataFrame({
    "Baseline SVM" : [acc_svm_b, pre_svm_b, rec_svm_b, f1_svm_b, auc_svm_b],
    "Tuned SVM"    : [acc_svm_t, pre_svm_t, rec_svm_t, f1_svm_t, auc_svm_t],
    "Baseline RF"  : [acc_rf_b,  pre_rf_b,  rec_rf_b,  f1_rf_b,  auc_rf_b],
    "Tuned RF"     : [acc_rf_t,  pre_rf_t,  rec_rf_t,  f1_rf_t,  auc_rf_t],
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

df_compare = df_compare.round(4)
print("\n-- Perbandingan Baseline vs Tuned --")
print(df_compare.to_string())

# Pilih model terbaik dari SEMUA 4 model berdasarkan ROC-AUC
# ROC-AUC dipilih karena lebih informatif daripada accuracy untuk kelas tidak seimbang
best_label = df_compare.loc['ROC-AUC'].idxmax()
print(f"\nModel terbaik berdasarkan ROC-AUC: {best_label}")

# =============================================================================
# 9. SIMPAN MODEL & ARTEFAK
# =============================================================================

SAVE_DIR = "Save_model"
os.makedirs(SAVE_DIR, exist_ok= True)

# Tentukan objek mode; & data test yang sesuai berdasarkan best_label
model_map = {
    "Baseline SVM"  : (svm_base, X_test_sc),
    "Tuned SVM"     : (search_svm, X_test_sc),
    "Baseline RF"   : (rf_base, X_test_enc),
    "Tuned RF"      : (search_rf, X_test_enc)
}
best_model, _ = model_map[best_label]

#Simpan model terbaik
joblib.dump(best_model, os.path.join(SAVE_DIR, "best_model.pkl"))
print(f"\n[OK] Model terbaik ({best_label}) disimpan -> {SAVE_DIR}/best_model.pkl")

# Simpan scaler (selalu dibutuhkan untuk SVM; disimpan sekaligus untuk RF jaga-jaga)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
print(f"[OK] Scaler disimpan -> {SAVE_DIR}/scaler.pkl")

joblib.dump(preprocessor,  os.path.join(SAVE_DIR, "preprocessor.pkl")) 
print(f"[OK] preprocessor disimpan -> {SAVE_DIR}/preprocessor.pkl")

# Simpan nama fitur agar urutan kolom konsisten saat prediksi di Streamlit
feature_names = list(X.columns)
joblib.dump(feature_names, os.path.join(SAVE_DIR, "feature_names.pkl"))
print(f"[OK] Feature names disimpan -> {SAVE_DIR}/feature_names.pkl")

# Simpan label model terbaik (untuk referensi di Streamlit)
joblib.dump(best_label, os.path.join(SAVE_DIR, "best_label.pkl"))
print(f"[OK] Best label disimpan -> {SAVE_DIR}/best_label.pkl")

joblib.dump(clip_bounds,   os.path.join(SAVE_DIR, "clip_bounds.pkl")) 
print(f"[OK] clip_bounds disimpan -> {SAVE_DIR}/clip_bounds.pkl")

# Verifikasi - load ulang dan cek
_model_check = joblib.load(os.path.join(SAVE_DIR, "best_model.pkl"))
print(f"\n[CHECK] Verifikasi load ulang: {type(_model_check).__name__} - OK")

# =============================================================================
# 10. VISUALISASI
# =============================================================================

# Visualisasi 1: Confusion Matrix Baseline
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Confusion Matrix — Baseline SVM vs Baseline Random Forest',fontsize=13, fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_svm_base), annot=True, fmt='d',
            cmap='Blues', ax=axes1[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes1[0].set(title=f'Baseline SVM\nAcc={acc_svm_b:.3f} | AUC={auc_svm_b:.3f}',ylabel='Actual', xlabel='Predicted')

sns.heatmap(confusion_matrix(y_test, y_pred_rf_base), annot=True, fmt='d',
            cmap='Greens', ax=axes1[1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes1[1].set(title=f'Baseline Random Forest\nAcc={acc_rf_b:.3f} | AUC={auc_rf_b:.3f}',ylabel='Actual', xlabel='Predicted')

plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: baseline_confusion_matrix.png")

# Visualisasi 2: Confusion Matrix Tuned
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Confusion Matrix — Tuned SVM vs Tuned RF', fontsize=13, fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_svm_tuned), annot=True, fmt='d',
            cmap='Blues', ax=axes2[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes2[0].set(title=f'Tuned SVM\nAcc={acc_svm_t:.3f} | AUC={auc_svm_t:.3f}',
            ylabel='Actual', xlabel='Predicted')

sns.heatmap(confusion_matrix(y_test, y_pred_rf_tuned), annot=True, fmt='d',
            cmap='Greens', ax=axes2[1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes2[1].set(title=f'Tuned RF\nAcc={acc_rf_t:.3f} | AUC={auc_rf_t:.3f}',
            ylabel='Actual', xlabel='Predicted')

plt.tight_layout()
plt.savefig('tuned_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: tuned_confusion_matrix.png")

# Visualisasi 3: ROC Curve — Baseline vs Tuned (semua model)
plt.figure(figsize=(9, 6))
fpr_svm_b, tpr_svm_b, _ = roc_curve(y_test, y_prob_svm_base)
fpr_rf_b,  tpr_rf_b,  _ = roc_curve(y_test, y_prob_rf_base)
fpr_svm_t, tpr_svm_t, _ = roc_curve(y_test, y_prob_svm_tuned)
fpr_rf_t,  tpr_rf_t,  _ = roc_curve(y_test, y_prob_rf_tuned)

plt.plot(fpr_svm_b, tpr_svm_b, color='#a8c4e0', lw=1.5, linestyle='--',label=f'Baseline SVM  (AUC={auc_svm_b:.3f})')
plt.plot(fpr_rf_b,  tpr_rf_b,  color='#a8d5b5', lw=1.5, linestyle='--',label=f'Baseline RF   (AUC={auc_rf_b:.3f})')
plt.plot(fpr_svm_t, tpr_svm_t, color='#2e75b6', lw=2,label=f'Tuned SVM     (AUC={auc_svm_t:.3f})')
plt.plot(fpr_rf_t,  tpr_rf_t,  color='#375623', lw=2,label=f'Tuned RF      (AUC={auc_rf_t:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

plt.fill_between(fpr_svm_t, tpr_svm_t, alpha=0.07, color='#2e75b6')
plt.fill_between(fpr_rf_t,  tpr_rf_t,  alpha=0.07, color='#375623')
plt.title('ROC Curve — Baseline vs Tuned (SVM & RF)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve_all.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: roc_curve_all.png")

# Visualisasi 4: Bar Chart Perbandingan Semua Metrik
fig, ax = plt.subplots(figsize=(12, 5))
x_pos  = np.arange(len(df_compare.index))
width  = 0.2
colors = ['#a8c4e0', '#2e75b6', '#a8d5b5', '#375623']

for i, (col, color) in enumerate(zip(df_compare.columns, colors)):
    bars = ax.bar(x_pos + i * width, df_compare[col], width, label=col, color=color, alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(df_compare.index)
ax.set_ylim(0, 1.18)
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrik — Baseline vs Tuned (SVM & RF)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('comparison_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: comparison_metrics.png")

# =============================================================================

print("\n Selesai! File yang disimpan:")
print("   [Model]")
print("   -> Save_model/best_model.pkl")
print("   -> Save_model/scaler.pkl")
print("   -> Save_model/preprocessor.pkl")
print("   -> Save_model/feature_names.pkl")
print("   -> Save_model/best_label.pkl")
print("   -> Save_model/clip_bounds.pkl")
print("   [Visualisasi]")
print("   -> baseline_confusion_matrix.png")
print("   -> tuned_confusion_matrix.png")
print("   -> roc_curve_all.png")
print("   -> comparison_metrics.png")
