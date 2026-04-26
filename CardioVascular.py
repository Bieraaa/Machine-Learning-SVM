import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
 
# ── Visualisasi 3: Bar Chart perbandingan metrik ─────────────────
plt.figure(figsize=(10, 6))
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
svm_scores   = [acc_svm, pre_svm, rec_svm, f1_svm, auc_svm]
rf_scores    = [acc_rf,  pre_rf,  rec_rf,  f1_rf,  auc_rf]
x            = np.arange(len(metric_names))
width        = 0.35
 
bars1 = plt.bar(x - width/2, svm_scores, width, label='Baseline SVM',
                color='steelblue', alpha=0.85)
bars2 = plt.bar(x + width/2, rf_scores,  width, label='Baseline RF',
                color='seagreen',  alpha=0.85)
 
plt.ylim(0, 1.15)
plt.xticks(x, metric_names)
plt.title('Perbandingan Metrik — Baseline SVM vs Random Forest')
plt.ylabel('Score')
plt.legend()
 
for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
 
plt.tight_layout()
plt.savefig('baseline_metric_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
 
print("\nSelesai! File disimpan:")
print("  → baseline_confusion_matrix.png")
print("  → baseline_roc_curve.png")
print("  → baseline_metric_comparison.png")