import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
<<<<<<< HEAD
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, roc_curve,confusion_matrix, classification_report)

""" ================= Load Data =================  """
df = pd.read_csv("heart.csv")
=======
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)

# ================= 1. Load Data =================
print("Memuat Data...")
# Pastikan path ini benar sesuai dengan lokasi file di komputermu
df = pd.read_csv(r"D:\itk\semester 4\ml\project tubes\Machine-Learning-SVM\heart.csv")
>>>>>>> 5a2151877824819f53c7450a03bcf26077948008
print(f"Shape awal: {df.shape}\n")
print(df.head()) 

# ================= 2. Cleansing Data =================
print("\nMelakukan Data Cleansing...")
df.drop_duplicates(inplace=True)

for col in ['Cholesterol', 'RestingBP']:
    median = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median)

num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for col in num_cols:
    df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

print(f"Shape setelah cleansing: {df.shape}")

<<<<<<< HEAD
""" ================= Pra-Proses Data =================  """

# Encoding kategorikal —  One-Hot Encoding
df = pd.get_dummies(df,columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],drop_first=True)

# trainsplit
X = df.drop(columns='HeartDisease')
=======
# ================= 3. Split Fitur & Target =================
X = df.drop(columns=['HeartDisease'])
>>>>>>> 5a2151877824819f53c7450a03bcf26077948008
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)

<<<<<<< HEAD
""" ================= Scaler =================  """
# Scaling — wajib untuk SVM
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
=======
# ================= 4. Konfigurasi Pipeline =================
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numeric_features = [col for col in X.columns if col not in categorical_features]
>>>>>>> 5a2151877824819f53c7450a03bcf26077948008

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

<<<<<<< HEAD
""" ================= Baseline model =================  """
# --- Baseline SVM ---
svm_base = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_base.fit(X_train_sc, y_train)
print("\nBaseline SVM selesai dilatih.")

# --- Baseline Random Forest ---
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
print("Baseline Random Forest selesai dilatih.")

""" ================= Hyperparamater Search Space =================  """
# SVM: C (keketatan batas), gamma (pengaruh data point), kernel (bentuk pemisah)

parameter_SVM = {
    'C' : [0.1, 1, 5, 10, 50, 100],
    'gamma' : ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel' : ['rbf', 'linear'],
    'class_weight': ['balanced', None]
}

# RF: n_estimators (jumlah pohon), max_depth (kedalaman), dll

parameter_RF = {
    'n_estimators' : [100, 200, 300, 500],
    'max_depth' : [None, 5, 10, 15, 20],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features' : ['sqrt', 'log2'],
    'class_weight'     : ['balanced', None]
}

#---------------- Randomized Search -----------------------

search_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    parameter_SVM, n_iter=60, cv= 10,
    scoring='recall', random_state=42, n_jobs=-1, refit=True
)

search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    parameter_RF, n_iter=70, cv=10,
    scoring= 'recall', random_state=42, n_jobs= -1, refit=True
)

#------------------- Tuning ------------------------------

print("Tuning SVM.... ", end='', flush=True)
search_svm.fit(X_train_sc, y_train)
print("Selesai... ")

print("Tuning RF... ", end='', flush=True)
search_rf.fit(X_train, y_train)
print("selesai... ")

print("\n -- Best paramater SVM -- ")
print(search_svm.best_params_)
print("\n -- Best parameter RF -- ")
print(search_rf.best_params_)

""" ================= Evaluasi =================  """

def evaluate_model(model, X_te, y_te, label, params):
=======
pipeline_svm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ================= 5. Hyperparameter Tuning =================
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 50, 100],
    'classifier__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear']
}

param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

print("\nMenyiapkan RandomizedSearchCV...")
rs_svm = RandomizedSearchCV(
    estimator=pipeline_svm,
    param_distributions=param_grid_svm,
    n_iter=15,          
    scoring='roc_auc',   # Memaksa mesin menyeimbangkan prediksi sakit & sehat
    cv=5,               
    random_state=42,
    n_jobs=-1           
)

rs_rf = RandomizedSearchCV(
    estimator=pipeline_rf,
    param_distributions=param_grid_rf,
    n_iter=15,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1
)

print("Memulai tuning Hyperparameter SVM... (Mohon tunggu sebentar)")
rs_svm.fit(X_train, y_train)
print("✅ Tuning SVM selesai!")

print("Memulai tuning Hyperparameter Random Forest... (Mohon tunggu sebentar)")
rs_rf.fit(X_train, y_train)
print("✅ Tuning RF selesai!")

best_svm_pipeline = rs_svm.best_estimator_
best_rf_pipeline = rs_rf.best_estimator_

# ================= 6. Evaluasi =================
def evaluate_baseline(model, X_te, y_te, label, params):
>>>>>>> 5a2151877824819f53c7450a03bcf26077948008
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    accuracy  = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred)
    recall    = recall_score(y_te, y_pred)
    f1        = f1_score(y_te, y_pred)
    auc       = roc_auc_score(y_te, y_prob)

    print("\n" + "=" * 45)
    print(f"  EVALUASI {label}")
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

<<<<<<< HEAD
# Evaluasi SVM
y_pred_svm_base, y_prob_svm_base, acc_svm_b, pre_svm_b, rec_svm_b, f1_svm_b, auc_svm_b = evaluate_model(
    svm_base, X_test_sc, y_test,
    label="BASELINE SVM",
    params={'Kernel': 'rbf', 'C': '1.0', 'Gamma': 'scale'}
)

# Evaluasi Random Forest
y_pred_rf_base, y_prob_rf_base, acc_rf_b, pre_rf_b, rec_rf_b, f1_rf_b, auc_rf_b = evaluate_model(
    rf_base, X_test, y_test,
    label="BASELINE Random Forest",
    params={'n_estimators': '100', 'max_depth': 'None', 'max_features': 'sqrt'}
)

# Tuned SVM
y_pred_svm_tuned, y_prob_svm_tuned, acc_svm_t, pre_svm_t, rec_svm_t, f1_svm_t, auc_svm_t = evaluate_model(
    search_svm, X_test_sc, y_test,
    label="TUNED SVM",
    params=search_svm.best_params_
)

# Tuned Random Forest
y_pred_rf_tuned, y_prob_rf_tuned, acc_rf_t, pre_rf_t, rec_rf_t, f1_rf_t, auc_rf_t = evaluate_model(
    search_rf, X_test, y_test,
    label="TUNED Random Forest",
    params=search_rf.best_params_
)

""" ================= Tabel Perbandingan =================  """

df_compare = pd.DataFrame({
    "Baseline SVM" : [acc_svm_b, pre_svm_b, rec_svm_b, f1_svm_b, auc_svm_b],
    "Tuned SVM"    : [acc_svm_t, pre_svm_t, rec_svm_t, f1_svm_t, auc_svm_t],
    "Baseline RF"  : [acc_rf_b,  pre_rf_b,  rec_rf_b,  f1_rf_b,  auc_rf_b],
    "Tuned RF"     : [acc_rf_t,  pre_rf_t,  rec_rf_t,  f1_rf_t,  auc_rf_t],
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

df_compare = df_compare.round(4)
print("\n-- Perbandingan Baseline vs Tuned --")
print(df_compare.to_string())

# ── Pilih model terbaik dari SEMUA 4 model (berdasarkan ROC-AUC) ──
best_label = df_compare.loc['ROC-AUC'].idxmax()
print(f"\nModel terbaik berdasarkan ROC-AUC: {best_label}")

""" ================= Simpan Model =================  """

SAVE_DIR = "Save_model"
os.makedirs(SAVE_DIR, exist_ok= True)

# Tentukan objek mode; & data test yang sesuai berdasarkan best_label
model_map = {
    "Baseline SVM"  : (svm_base, X_test_sc),
    "Tuned SVM"     : (search_svm, X_test_sc),
    "Baseline RF"   : (rf_base, X_test),
    "Tuned RF"      : (search_rf, X_test)
}
best_model, _ = model_map[best_label]

#Simpan model terbaik
joblib.dump(best_model, os.path.join(SAVE_DIR, "best_model.pkl"))
print(f"\n✅ Model terbaik ({best_label}) disimpan → {SAVE_DIR}/best_model.pkl")

# Simpan scaler (selalu dibutuhkan untuk SVM; disimpan sekaligus untuk RF jaga-jaga)
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
print(f"✅ Scaler disimpan → {SAVE_DIR}/scaler.pkl")

# Simpan nama fitur agar urutan kolom konsisten saat prediksi di Streamlit
feature_names = list(X.columns)
joblib.dump(feature_names, os.path.join(SAVE_DIR, "feature_names.pkl"))
print(f"✅ Feature names disimpan → {SAVE_DIR}/feature_names.pkl")

# Simpan label model terbaik (untuk referensi di Streamlit)
joblib.dump(best_label, os.path.join(SAVE_DIR, "best_label.pkl"))
print(f"✅ Best label disimpan → {SAVE_DIR}/best_label.pkl")

# Verifikasi — load ulang dan cek
_model_check = joblib.load(os.path.join(SAVE_DIR, "best_model.pkl"))
print(f"\n🔍 Verifikasi load ulang: {type(_model_check).__name__} — OK")

""" ================= Visualisasi =================  """
# ── Visualisasi 1: Confusion Matrix Baseline ──────────────────
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
plt.close()
print("Saved: baseline_confusion_matrix.png")

# ── Visualisasi 2: Confusion Matrix Tuned ────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Confusion Matrix — Tuned SVM vs Tuned Random Forest',fontsize=13, fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_svm_tuned), annot=True, fmt='d',
            cmap='Blues', ax=axes2[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes2[0].set(title=f'Tuned SVM\nAcc={acc_svm_t:.3f} | AUC={auc_svm_t:.3f}',ylabel='Actual', xlabel='Predicted')

sns.heatmap(confusion_matrix(y_test, y_pred_rf_tuned), annot=True, fmt='d',
            cmap='Greens', ax=axes2[1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes2[1].set(title=f'Tuned RF\nAcc={acc_rf_t:.3f} | AUC={auc_rf_t:.3f}',ylabel='Actual', xlabel='Predicted')

plt.tight_layout()
plt.savefig('tuned_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: tuned_confusion_matrix.png")

# ── Visualisasi 3: ROC Curve — Baseline vs Tuned (semua model) ─
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
=======
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

# ================= 7. Visualisasi =================
print("\nMenyiapkan grafik... (PENTING: Tutup jendela grafik (tanda X) untuk memunculkan grafik selanjutnya!)")

# ── Visualisasi 1: Confusion Matrix berdampingan ─────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Confusion Matrix — Tuned SVM vs Random Forest', fontsize=13, fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d',
            cmap='Blues', ax=axes1[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes1[0].set(title=f'Tuned SVM\nAcc={acc_svm:.3f} | AUC={auc_svm:.3f}', ylabel='Actual', xlabel='Predicted')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
            cmap='Greens', ax=axes1[1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes1[1].set(title=f'Tuned Random Forest\nAcc={acc_rf:.3f} | AUC={auc_rf:.3f}', ylabel='Actual', xlabel='Predicted')

plt.tight_layout()
plt.show()

# ── Visualisasi 2: ROC Curve gabungan ────────────────────────────
plt.figure(figsize=(8, 6))
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_svm, tpr_svm, color='steelblue', lw=2, label=f'Tuned SVM (AUC={auc_svm:.3f})')
plt.plot(fpr_rf,  tpr_rf,  color='seagreen',  lw=2, label=f'Tuned RF  (AUC={auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
plt.fill_between(fpr_svm, tpr_svm, alpha=0.05, color='steelblue')
plt.fill_between(fpr_rf,  tpr_rf,  alpha=0.05, color='seagreen')
plt.title('ROC Curve — Tuned SVM vs Random Forest')
>>>>>>> 5a2151877824819f53c7450a03bcf26077948008
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
<<<<<<< HEAD
plt.savefig('roc_curve_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curve_all.png")

# ── Visualisasi 4: Bar Chart Perbandingan Semua Metrik ────────
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
plt.close()
print("Saved: comparison_metrics.png")

# ── Visualisasi 5: Confusion Matrix Tuned berdampingan ───────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Confusion Matrix — Tuned SVM vs Tuned Random Forest', fontsize=13, fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_svm_tuned), annot=True, fmt='d',
            cmap='Blues', ax=axes2[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes2[0].set(title=f'Tuned SVM\nAcc={df_compare.loc["Accuracy","Tuned SVM"]:.3f} | AUC={df_compare.loc["ROC-AUC","Tuned SVM"]:.3f}', ylabel='Actual', xlabel='Predicted')

sns.heatmap(confusion_matrix(y_test, y_pred_rf_tuned), annot=True, fmt='d',
            cmap='Greens', ax=axes2[1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes2[1].set(title=f'Tuned RF\nAcc={df_compare.loc["Accuracy","Tuned RF"]:.3f} | AUC={df_compare.loc["ROC-AUC","Tuned RF"]:.3f}', ylabel='Actual', xlabel='Predicted')

plt.tight_layout()
plt.savefig('tuned_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

""" =========================================================== """

print("\n Selesai! File yang disimpan:")
print("   [Model]")
print("   → saved_model/best_model.pkl")
print("   → saved_model/scaler.pkl")
print("   → saved_model/feature_names.pkl")
print("   → saved_model/best_label.pkl")
print("   [Visualisasi]")
print("   → baseline_confusion_matrix.png")
print("   → tuned_confusion_matrix.png")
print("   → roc_curve_all.png")
print("   → comparison_metrics.png")
=======
plt.show()

# ── Visualisasi 3: Bar Chart perbandingan metrik ─────────────────
plt.figure(figsize=(10, 6))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
svm_scores   = [acc_svm, pre_svm, rec_svm, f1_svm, auc_svm]
rf_scores    = [acc_rf,  pre_rf,  rec_rf,  f1_rf,  auc_rf]
x            = np.arange(len(metric_names))
width        = 0.35

bars1 = plt.bar(x - width/2, svm_scores, width, label='Tuned SVM',
                color='steelblue', alpha=0.85)
bars2 = plt.bar(x + width/2, rf_scores,  width, label='Tuned RF',
                color='seagreen',  alpha=0.85)

plt.ylim(0, 1.15)
plt.xticks(x, metric_names)
plt.title('Perbandingan Metrik — Tuned SVM vs Random Forest')
plt.ylabel('Score')
plt.legend()

for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ================= 8. Save Model =================
joblib.dump(best_svm_pipeline, 'model_svm_jantung.joblib')
joblib.dump(best_rf_pipeline, 'model_rf_jantung.joblib')

print("\nKodingan selesai dieksekusi 100%! Model berhasil disimpan ke format .joblib")
>>>>>>> 5a2151877824819f53c7450a03bcf26077948008
