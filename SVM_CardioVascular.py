import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform, randint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

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
df = pd.get_dummies(df,
     columns=['Sex','ChestPainType','RestingECG',
              'ExerciseAngina','ST_Slope'],
     drop_first=True)

# Split fitur & target
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

""" ================= Tuning model =================  """
# --- SVM ---
print("\nTuning SVM...")
search_svm = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_distributions={
        'C'     : loguniform(0.01, 1000),
        'gamma' : loguniform(0.001, 10),
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    },
    n_iter=100, cv=10, scoring='roc_auc', n_jobs=-1, random_state=42
)
search_svm.fit(X_train_sc, y_train)
best_svm = search_svm.best_estimator_
print(f"SVM  Best params : {search_svm.best_params_}")
print(f"SVM  Best CV AUC : {search_svm.best_score_:.4f}")
 
# --- Random Forest ---
# RF tidak memerlukan scaling, gunakan data asli
print("\nTuning Random Forest...")
search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions={
        'n_estimators'     : randint(50, 300),
        'max_depth'        : [None, 5, 10, 20],
        'min_samples_split': randint(2, 10),
        'max_features'     : ['sqrt', 'log2']
    },
    n_iter=100, cv=10, scoring='roc_auc', n_jobs=-1, random_state=42
)
search_rf.fit(X_train, y_train)
best_rf = search_rf.best_estimator_
print(f"RF   Best params : {search_rf.best_params_}")
print(f"RF   Best CV AUC : {search_rf.best_score_:.4f}")

""" ================= Evaluasi =================  """

def evaluate(model, X_test, y_test, X_train, y_train, label):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv     = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
    metrics = {
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall'   : recall_score(y_test, y_pred),
        'F1-Score' : f1_score(y_test, y_pred),
        'ROC-AUC'  : roc_auc_score(y_test, y_prob),
        'CV AUC'   : cv.mean()
    }
    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Heart Disease'])}")
    return y_pred, y_prob, metrics
 
y_pred_svm, y_prob_svm, metrics_svm = evaluate(
    best_svm, X_test_sc, y_test, X_train_sc, y_train, "SVM"
)
y_pred_rf, y_prob_rf, metrics_rf = evaluate(
    best_rf, X_test, y_test, X_train, y_train, "Random Forest"
)

""" ================= Visualisasi =================  """

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Perbandingan SVM vs Random Forest — Heart Failure Prediction',
             fontsize=14, fontweight='bold', y=1.01)
 
# ── Confusion Matrix SVM ─────────────────────────────────────────
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d',
            cmap='Blues', ax=axes[0][0],
            xticklabels=['Normal','Heart Disease'],
            yticklabels=['Normal','Heart Disease'])
axes[0][0].set(title='Confusion Matrix — SVM',
               ylabel='Actual', xlabel='Predicted')
 
# ── Confusion Matrix RF ──────────────────────────────────────────
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
            cmap='Greens', ax=axes[0][1],
            xticklabels=['Normal','Heart Disease'],
            yticklabels=['Normal','Heart Disease'])
axes[0][1].set(title='Confusion Matrix — Random Forest',
               ylabel='Actual', xlabel='Predicted')
 
# ── ROC Curve (keduanya dalam satu grafik) ───────────────────────
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
axes[0][2].plot(fpr_svm, tpr_svm, color='steelblue', lw=2,
                label=f"SVM  (AUC={metrics_svm['ROC-AUC']:.3f})")
axes[0][2].plot(fpr_rf,  tpr_rf,  color='seagreen',  lw=2,
                label=f"RF   (AUC={metrics_rf['ROC-AUC']:.3f})")
axes[0][2].plot([0,1],[0,1], 'k--', lw=1)
axes[0][2].fill_between(fpr_svm, tpr_svm, alpha=0.05, color='steelblue')
axes[0][2].fill_between(fpr_rf,  tpr_rf,  alpha=0.05, color='seagreen')
axes[0][2].set(title='ROC Curve — SVM vs Random Forest',
               xlabel='False Positive Rate', ylabel='True Positive Rate')
axes[0][2].legend()
 
# ── Bar Chart Perbandingan Metrik ────────────────────────────────
metric_names  = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
svm_scores    = [metrics_svm[m] for m in metric_names]
rf_scores     = [metrics_rf[m]  for m in metric_names]
x             = np.arange(len(metric_names))
width         = 0.35
 
bars1 = axes[1][0].bar(x - width/2, svm_scores, width, label='SVM',
                        color='steelblue', alpha=0.85)
bars2 = axes[1][0].bar(x + width/2, rf_scores,  width, label='Random Forest',
                        color='seagreen',  alpha=0.85)
axes[1][0].set_ylim(0, 1.15)
axes[1][0].set_xticks(x)
axes[1][0].set_xticklabels(metric_names, rotation=15)
axes[1][0].set(title='Perbandingan Metrik Evaluasi', ylabel='Score')
axes[1][0].legend()
for bar in bars1:
    axes[1][0].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    axes[1][0].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
 
# ── Feature Importance RF ────────────────────────────────────────
feat_imp = pd.Series(best_rf.feature_importances_,
                     index=X.columns).sort_values(ascending=True).tail(10)
feat_imp.plot(kind='barh', ax=axes[1][1], color='seagreen', alpha=0.8)
axes[1][1].set(title='Feature Importance — Random Forest',
               xlabel='Importance Score')
 
# ── Tabel Ringkasan ──────────────────────────────────────────────
axes[1][2].axis('off')
table_data = [[m,
               f"{metrics_svm[m]:.4f}",
               f"{metrics_rf[m]:.4f}",
               '✓ SVM' if metrics_svm[m] > metrics_rf[m] else '✓ RF']
              for m in metric_names]
table_data.append(['CV AUC',
                   f"{metrics_svm['CV AUC']:.4f}",
                   f"{metrics_rf['CV AUC']:.4f}",
                   '✓ SVM' if metrics_svm['CV AUC'] > metrics_rf['CV AUC'] else '✓ RF'])
 
tbl = axes[1][2].table(
    cellText=table_data,
    colLabels=['Metrik', 'SVM', 'Random Forest', 'Unggul'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2)
axes[1][2].set_title('Ringkasan Perbandingan', fontweight='bold', pad=20)
 
plt.tight_layout()
plt.savefig('svm_vs_rf.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSelesai! File disimpan: svm_vs_rf.png")