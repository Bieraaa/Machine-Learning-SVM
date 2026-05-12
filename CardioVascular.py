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
df = pd.read_csv("D:\Tubes Machine Learning\heart.csv")
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

# Scaling — wajib untuk SVM
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")

""" ================= model =================  """
# --- Baseline SVM ---
svm_base = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_base.fit(X_train_sc, y_train)
print("\nBaseline SVM selesai dilatih.")

# --- Baseline Random Forest ---
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
print("Baseline Random Forest selesai dilatih.")

#-------Search Space-------------
# SVM: C (keketatan batas), gamma (pengaruh data point), kernel (bentuk pemisah)

parameter_SVM = {
    'C' : [0.01, 0.1, 1, 10, 100, 500, 1000],
    'gamma' : ['scale', 'auto', 0.01, 0.1],
    'kernel' : ['rbf', 'linear'],
}

# RF: n_estimators (jumlah pohon), max_depth (kedalaman), dll

parameter_RF = {
    'n_estimators' : [100, 200, 300],
    'max_depth' : [None, 5, 10, 20],
    'min_samples_split' : [2, 5, 10],
    'max_features' : ['sqrt', 'log2']
}

#---------------- Randomized Search -----------------------

search_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    parameter_SVM, n_iter=50, cv= 5,
    scoring='roc_auc', random_state=42, n_jobs=-1, refit=True
)

search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    parameter_RF, n_iter=50, cv=5,
    scoring= 'roc_auc', random_state=42, n_jobs= -1, refit=True
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

def evaluate_baseline(model, X_te, y_te, label, params):
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
y_pred_svm, y_prob_svm, acc_svm, pre_svm, rec_svm, f1_svm, auc_svm = evaluate_baseline(
    svm_base, X_test_sc, y_test,
    label="SVM",
    params={'Kernel': 'rbf (default)', 'C': '1.0 (default)', 'Gamma': 'scale (default)'}
)

# Evaluasi Random Forest
y_pred_rf, y_prob_rf, acc_rf, pre_rf, rec_rf, f1_rf, auc_rf = evaluate_baseline(
    rf_base, X_test, y_test,
    label="Random Forest",
    params={'n_estimators': '100 (default)', 'max_depth': 'None (default)', 'max_features': 'sqrt (default)'}
)

""" ================= Visualisasi =================  """
# ── Visualisasi 1: Confusion Matrix berdampingan ─────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle('Confusion Matrix — Baseline SVM vs Random Forest', fontsize=13, fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d',
            cmap='Blues', ax=axes1[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes1[0].set(title=f'Baseline SVM\nAcc={acc_svm:.3f} | AUC={auc_svm:.3f}', ylabel='Actual', xlabel='Predicted')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
            cmap='Greens', ax=axes1[1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes1[1].set(title=f'Baseline Random Forest\nAcc={acc_rf:.3f} | AUC={auc_rf:.3f}', ylabel='Actual', xlabel='Predicted')

plt.tight_layout()
plt.savefig('baseline_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Visualisasi 2: ROC Curve gabungan ────────────────────────────
plt.figure(figsize=(8, 6))
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)

plt.plot(fpr_svm, tpr_svm, color='steelblue', lw=2, label=f'Baseline SVM (AUC={auc_svm:.3f})')
plt.plot(fpr_rf,  tpr_rf,  color='seagreen',  lw=2, label=f'Baseline RF  (AUC={auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
plt.fill_between(fpr_svm, tpr_svm, alpha=0.05, color='steelblue')
plt.fill_between(fpr_rf,  tpr_rf,  alpha=0.05, color='seagreen')
plt.title('ROC Curve — Baseline SVM vs Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('baseline_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

#visualisasi 3 : Tabel & Bar Chart Baseline vs Tuned 
y_pred_svm_tuned = search_svm.predict(X_test_sc)
y_pred_rf_tuned = search_rf.predict(X_test)
y_prob_svm_tuned = search_svm.predict_proba(X_test_sc)[:, 1]
y_prob_rf_tuned = search_rf.predict_proba(X_test)[:, 1]

df_compare = pd.DataFrame({
    "Baseline SVM" : [acc_svm, pre_svm, rec_svm, f1_svm, auc_svm],
    "Tuned SVM" : [ 
        accuracy_score(y_test, y_pred_svm_tuned),
        precision_score(y_test, y_pred_svm_tuned),
        recall_score(y_test, y_pred_svm_tuned),
        f1_score(y_test, y_pred_svm_tuned),
        roc_auc_score(y_test, y_prob_svm_tuned),
    ],
    "Baseline RF" : [acc_rf, pre_rf, rec_rf, f1_rf, auc_rf],
    "Tuned RF" : [
        accuracy_score(y_test, y_pred_rf_tuned),
        precision_score(y_test, y_pred_rf_tuned),
        recall_score(y_test, y_pred_rf_tuned),
        f1_score(y_test, y_pred_rf_tuned),
        roc_auc_score(y_test, y_prob_rf_tuned),
    ],
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

df_compare = df_compare.round(4)
print("\n -- perbandingan baseline dan tuning")
print(df_compare.to_string())

#pilih model terbaik berdasarkan ROC AUC
if df_compare.loc['ROC-AUC', 'Tuned SVM'] >= df_compare.loc['ROC-AUC', 'Tuned RF']:
    best_label = "Tuned SVM"
    y_pred_best = y_pred_svm_tuned
else:
    best_label = "Tuned RF"
    y_pred_best = y_pred_rf_tuned

print(f"\n Model terbaik : {best_label}")

#Bar Chart Baseline vs Tuned
fig, ax = plt.subplots(figsize=(11, 5))
x_pos  = np.arange(len(df_compare.index))
width  = 0.2
colors = ['#5b9bd5', '#2e75b6', '#70ad47', '#375623']

for i, (col, color) in enumerate(zip(df_compare.columns, colors)):
    bars = ax.bar(x_pos + i*width, df_compare[col],width, label=col, color=color, alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x_pos + width*1.5)
ax.set_xticklabels(df_compare.index)
ax.set_ylim(0, 1.18)
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrik — Baseline vs Tuned (SVM & RF)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('comparison_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

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