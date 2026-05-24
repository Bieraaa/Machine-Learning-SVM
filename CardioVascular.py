import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, roc_curve,confusion_matrix, classification_report)

""" ================= Load Data =================  """
df = pd.read_csv("heart.csv")
print(f"Shape awal: {df.shape}\n")
print(df.head())

""" ================= Cleansing Data =================  """

# Hapus duplikat
df.drop_duplicates(inplace=True)

# Nilai 0 tidak valid → ganti median
for col in ['Cholesterol', 'RestingBP']:
    median = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median)

# Winsorize outlier (1%–99%)
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for col in num_cols:
    df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

print(f"Shape setelah cleansing: {df.shape}")

""" ================= Pra-Proses Data =================  """

# Encoding kategorikal —  One-Hot Encoding
df = pd.get_dummies(df,columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],drop_first=True)

# trainsplit
X = df.drop(columns='HeartDisease')
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)

""" ================= Scaler =================  """
# Scaling — wajib untuk SVM
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")

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
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    accuracy  = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred)
    recall    = recall_score(y_te, y_pred)
    f1        = f1_score(y_te, y_pred)
    auc       = roc_auc_score(y_te, y_prob)

    print("\n" + "=" * 45)
    print(f"  BASELINE {label}")
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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
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

print("\nSelesai! File disimpan:")
print("  → baseline_confusion_matrix.png")
print("  → comparison_metrics.png")
print("  → baseline_roc_curve.png")
print("  → tuned_confusion_matrix.png")