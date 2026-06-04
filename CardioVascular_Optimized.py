# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      GridSearchCV, RandomizedSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             make_scorer)
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.calibration import CalibratedClassifierCV

# ================= 1. Load Data =================
print("=" * 60)
print("  OPTIMIZED CARDIOVASCULAR PREDICTION PIPELINE")
print("=" * 60)
print("\n[1/8] Memuat Data...")
df = pd.read_csv(r"D:\itk\semester 4\ml\project tubes\Machine-Learning-SVM\heart.csv")
print(f"Shape awal: {df.shape}")
print(f"Distribusi target:\n{df['HeartDisease'].value_counts()}")
print(f"Rasio kelas: {df['HeartDisease'].value_counts(normalize=True).round(3).to_dict()}")

# ================= 2. Cleansing Data =================
print("\n[2/8] Data Cleansing & Feature Engineering...")
df.drop_duplicates(inplace=True)

# Ganti nilai 0 yang tidak valid dengan median
for col in ['Cholesterol', 'RestingBP']:
    median = df.loc[df[col] != 0, col].median()
    df[col] = df[col].replace(0, median)
    print(f"  - Kolom '{col}': nilai 0 diganti median ({median:.1f})")

# Winsorizing outlier (1%-99%)
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for col in num_cols:
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = df[col].clip(q01, q99)

# ================= 3. Feature Engineering =================
print("\n[3/8] Feature Engineering...")

# Fitur interaksi yang bermakna secara medis
df['Age_MaxHR_ratio']    = df['Age'] / (df['MaxHR'] + 1)          # Rasio usia vs detak jantung max
df['Chol_Age_product']   = df['Cholesterol'] * df['Age'] / 1000   # Produk kolesterol & usia
df['BP_age_ratio']       = df['RestingBP'] / df['Age']             # Tekanan darah relatif usia
df['MaxHR_Age_diff']     = (220 - df['Age']) - df['MaxHR']         # Defisit detak jantung max prediksi
df['Oldpeak_sq']         = df['Oldpeak'] ** 2                      # Transformasi kuadrat Oldpeak
df['is_elderly']         = (df['Age'] >= 60).astype(int)           # Flag lansia
df['high_chol']          = (df['Cholesterol'] >= 200).astype(int)  # Flag kolesterol tinggi
df['exercise_capacity']  = df['MaxHR'] / (df['Age'] + 1)          # Kapasitas latihan

print(f"  Fitur baru ditambahkan: 8 fitur rekayasa")
print(f"  Total fitur: {df.shape[1] - 1}")

# ================= 4. Split Fitur & Target =================
print("\n[4/8] Membagi Data...")
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# Stratified split untuk menjaga proporsi kelas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)
print(f"  Train: {X_train.shape[0]} sampel | Test: {X_test.shape[0]} sampel")
print(f"  Distribusi train: {dict(y_train.value_counts())}")
print(f"  Distribusi test : {dict(y_test.value_counts())}")

# ================= 5. Konfigurasi Pipeline =================
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numeric_features = [col for col in X.columns if col not in categorical_features]

# Preprocessor: RobustScaler lebih tahan outlier daripada StandardScaler
preprocessor_robust = ColumnTransformer(transformers=[
    ('num', RobustScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
])

preprocessor_std = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
])

# ================= 6. Hyperparameter Tuning - SVM =================
print("\n[5/8] Hyperparameter Tuning SVM (diperluas)...")

# SVM dengan class_weight='balanced' untuk atasi ketidakseimbangan kelas
pipeline_svm = Pipeline(steps=[
    ('preprocessor', preprocessor_robust),
    ('classifier', SVC(probability=True, random_state=42, class_weight='balanced'))
])

# Grid parameter lebih luas
param_grid_svm = {
    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'classifier__C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
    'classifier__degree': [2, 3],       # untuk kernel poly
}

# Scoring: F1 lebih baik daripada hanya AUC untuk klasifikasi klinis
f1_scorer = make_scorer(f1_score, average='binary')

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

rs_svm = RandomizedSearchCV(
    estimator=pipeline_svm,
    param_distributions=param_grid_svm,
    n_iter=50,               # Lebih banyak iterasi
    scoring='roc_auc',       # AUC untuk ranking probabilitas
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

rs_svm.fit(X_train, y_train)
print(f"  [OK] Best params SVM: {rs_svm.best_params_}")
print(f"  [OK] Best CV AUC: {rs_svm.best_score_:.4f}")

# ================= 7. Hyperparameter Tuning - Random Forest =================
print("\n[6/8] Hyperparameter Tuning Random Forest (diperluas)...")

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor_std),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced_subsample'))
])

# Grid parameter lebih lengkap
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [None, 10, 15, 20, 30],
    'classifier__min_samples_split': [2, 4, 5, 8, 10],
    'classifier__min_samples_leaf': [1, 2, 3, 4],
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__bootstrap': [True, False],
}

rs_rf = RandomizedSearchCV(
    estimator=pipeline_rf,
    param_distributions=param_grid_rf,
    n_iter=60,               # Lebih banyak iterasi
    scoring='roc_auc',
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True
)

rs_rf.fit(X_train, y_train)
print(f"  [OK] Best params RF: {rs_rf.best_params_}")
print(f"  [OK] Best CV AUC: {rs_rf.best_score_:.4f}")

best_svm_pipeline = rs_svm.best_estimator_
best_rf_pipeline  = rs_rf.best_estimator_

# ================= 8. Threshold Optimization untuk SVM =================
print("\n[7/8] Threshold Optimization (SVM)...")

# Cari threshold optimal yang memaksimalkan F1-score pada data latih
y_prob_train_svm = best_svm_pipeline.predict_proba(X_train)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores_thres = []

for thres in thresholds:
    y_pred_thres = (y_prob_train_svm >= thres).astype(int)
    f1_scores_thres.append(f1_score(y_train, y_pred_thres))

best_threshold_svm = thresholds[np.argmax(f1_scores_thres)]
print(f"  [OK] Optimal threshold SVM: {best_threshold_svm:.2f} (F1 train: {max(f1_scores_thres):.4f})")

# Lakukan juga untuk RF
y_prob_train_rf = best_rf_pipeline.predict_proba(X_train)[:, 1]
f1_scores_thres_rf = []

for thres in thresholds:
    y_pred_thres = (y_prob_train_rf >= thres).astype(int)
    f1_scores_thres_rf.append(f1_score(y_train, y_pred_thres))

best_threshold_rf = thresholds[np.argmax(f1_scores_thres_rf)]
print(f"  [OK] Optimal threshold RF: {best_threshold_rf:.2f} (F1 train: {max(f1_scores_thres_rf):.4f})")

# ================= 9. Evaluasi =================
print("\n[8/8] Evaluasi Final...")

def evaluate_model(model, X_te, y_te, label, params, threshold=0.5):
    """Evaluasi komprehensif dengan threshold kustom."""
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)   # Pakai threshold optimal

    accuracy  = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall    = recall_score(y_te, y_pred, zero_division=0)
    f1        = f1_score(y_te, y_pred, zero_division=0)
    auc       = roc_auc_score(y_te, y_prob)

    # Cross-validation score pada test set (indikasi generalisasi)
    cv_auc = cross_val_score(model, X_te, y_te, cv=5, scoring='roc_auc').mean()

    print("\n" + "=" * 55)
    print(f"  EVALUASI {label}")
    print(f"  Threshold: {threshold:.2f}")
    print("=" * 55)
    for k, v in params.items():
        k_short = k.replace('classifier__', '')
        print(f"  {k_short:<20}: {v}")
    print("-" * 55)
    print(f"  Accuracy      : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  CV-AUC (5fld) : {cv_auc:.4f}")
    print("=" * 55)
    print(f"\nClassification Report — {label}:")
    print(classification_report(y_te, y_pred,
                                 target_names=['Normal', 'Heart Disease'],
                                 zero_division=0))

    return y_pred, y_prob, accuracy, precision, recall, f1, auc

# Evaluasi SVM dengan threshold optimal
y_pred_svm, y_prob_svm, acc_svm, pre_svm, rec_svm, f1_svm, auc_svm = evaluate_model(
    best_svm_pipeline, X_test, y_test,
    label="Optimized SVM",
    params=rs_svm.best_params_,
    threshold=best_threshold_svm
)

# Evaluasi RF dengan threshold optimal
y_pred_rf, y_prob_rf, acc_rf, pre_rf, rec_rf, f1_rf, auc_rf = evaluate_model(
    best_rf_pipeline, X_test, y_test,
    label="Optimized Random Forest",
    params=rs_rf.best_params_,
    threshold=best_threshold_rf
)

# ================= 10. Ensemble Stacking =================
print("\n🔥 BONUS: Membuat Model Stacking (SVM + RF + GB)...")

# Preprocessor khusus untuk stacking
preprocessor_stack = ColumnTransformer(transformers=[
    ('num', RobustScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Base estimators yang sudah di-tune
estimators = [
    ('svm', Pipeline([
        ('pre', preprocessor_robust),
        ('clf', SVC(probability=True, random_state=42, class_weight='balanced',
                    **{k.replace('classifier__',''): v for k, v in rs_svm.best_params_.items()}))
    ])),
    ('rf', Pipeline([
        ('pre', preprocessor_std),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced_subsample',
                                        **{k.replace('classifier__',''): v for k, v in rs_rf.best_params_.items()}))
    ])),
]

# Meta-classifier
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, random_state=42, max_iter=1000),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)
y_pred_stack  = stacking_model.predict(X_test)
y_prob_stack  = stacking_model.predict_proba(X_test)[:, 1]
acc_stack     = accuracy_score(y_test, y_pred_stack)
f1_stack      = f1_score(y_test, y_pred_stack)
auc_stack     = roc_auc_score(y_test, y_prob_stack)
rec_stack     = recall_score(y_test, y_pred_stack)
pre_stack     = precision_score(y_test, y_pred_stack)

print(f"\n  EVALUASI Stacking Ensemble")
print(f"  Accuracy  : {acc_stack:.4f}")
print(f"  Precision : {pre_stack:.4f}")
print(f"  Recall    : {rec_stack:.4f}")
print(f"  F1-Score  : {f1_stack:.4f}")
print(f"  ROC-AUC   : {auc_stack:.4f}")
print(f"\nClassification Report — Stacking Ensemble:")
print(classification_report(y_test, y_pred_stack,
                             target_names=['Normal', 'Heart Disease']))

# ================= 11. Visualisasi =================
print("\nMenyiapkan visualisasi...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Optimized Model Evaluation — SVM vs RF vs Stacking',
             fontsize=15, fontweight='bold', y=0.98)

# --- 1. Confusion Matrix SVM ---
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes[0, 0].set(title=f'Optimized SVM\nAcc={acc_svm:.3f} | AUC={auc_svm:.3f}',
               ylabel='Actual', xlabel='Predicted')

# --- 2. Confusion Matrix RF ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes[0, 1].set(title=f'Optimized RF\nAcc={acc_rf:.3f} | AUC={auc_rf:.3f}',
               ylabel='Actual', xlabel='Predicted')

# --- 3. Confusion Matrix Stacking ---
cm_stack = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Purples', ax=axes[0, 2],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes[0, 2].set(title=f'Stacking Ensemble\nAcc={acc_stack:.3f} | AUC={auc_stack:.3f}',
               ylabel='Actual', xlabel='Predicted')

# --- 4. ROC Curve gabungan ---
fpr_svm,   tpr_svm,   _ = roc_curve(y_test, y_prob_svm)
fpr_rf,    tpr_rf,    _ = roc_curve(y_test, y_prob_rf)
fpr_stack, tpr_stack, _ = roc_curve(y_test, y_prob_stack)

axes[1, 0].plot(fpr_svm,   tpr_svm,   color='steelblue',  lw=2.5, label=f'SVM (AUC={auc_svm:.3f})')
axes[1, 0].plot(fpr_rf,    tpr_rf,    color='seagreen',   lw=2.5, label=f'RF  (AUC={auc_rf:.3f})')
axes[1, 0].plot(fpr_stack, tpr_stack, color='darkorchid', lw=2.5, label=f'Stacking (AUC={auc_stack:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
axes[1, 0].fill_between(fpr_svm,   tpr_svm,   alpha=0.07, color='steelblue')
axes[1, 0].fill_between(fpr_rf,    tpr_rf,    alpha=0.07, color='seagreen')
axes[1, 0].fill_between(fpr_stack, tpr_stack, alpha=0.07, color='darkorchid')
axes[1, 0].set(title='ROC Curve Comparison', xlabel='False Positive Rate',
               ylabel='True Positive Rate')
axes[1, 0].legend(loc='lower right')

# --- 5. Bar Chart Metrik ---
metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
svm_scores    = [acc_svm, pre_svm, rec_svm, f1_svm, auc_svm]
rf_scores     = [acc_rf,  pre_rf,  rec_rf,  f1_rf,  auc_rf]
stack_scores  = [acc_stack, pre_stack, rec_stack, f1_stack, auc_stack]
x             = np.arange(len(metric_names))
width         = 0.28

b1 = axes[1, 1].bar(x - width, svm_scores,   width, label='Optimized SVM',
                    color='steelblue',  alpha=0.85)
b2 = axes[1, 1].bar(x,          rf_scores,    width, label='Optimized RF',
                    color='seagreen',   alpha=0.85)
b3 = axes[1, 1].bar(x + width,  stack_scores, width, label='Stacking',
                    color='darkorchid', alpha=0.85)
axes[1, 1].set_ylim(0, 1.20)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metric_names)
axes[1, 1].set_title('Metric Comparison')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()

for bar in [*b1, *b2, *b3]:
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7)

# --- 6. Feature Importance RF ---
# Ambil nama fitur setelah preprocessing
try:
    ohe = best_rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feat_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feat_names = numeric_features + cat_feat_names
    importances = best_rf_pipeline.named_steps['classifier'].feature_importances_

    feat_df = pd.DataFrame({'feature': all_feat_names, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=True).tail(15)

    axes[1, 2].barh(feat_df['feature'], feat_df['importance'],
                    color='seagreen', alpha=0.85)
    axes[1, 2].set_title('Top 15 Feature Importances (RF)')
    axes[1, 2].set_xlabel('Importance')
except Exception as e:
    axes[1, 2].text(0.5, 0.5, f'Feature importance\ntidak tersedia:\n{str(e)[:50]}',
                    ha='center', va='center', transform=axes[1, 2].transAxes)

plt.tight_layout()
plt.savefig('optimized_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# ================= 12. Ringkasan Perbandingan =================
print("\n" + "=" * 65)
print("  📋 RINGKASAN PERBANDINGAN PERFORMA")
print("=" * 65)
print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7} {'AUC':>7}")
print("-" * 65)
print(f"{'Optimized SVM':<25} {acc_svm:>9.4f} {pre_svm:>10.4f} {rec_svm:>7.4f} {f1_svm:>7.4f} {auc_svm:>7.4f}")
print(f"{'Optimized RF':<25} {acc_rf:>9.4f} {pre_rf:>10.4f} {rec_rf:>7.4f} {f1_rf:>7.4f} {auc_rf:>7.4f}")
print(f"{'Stacking Ensemble':<25} {acc_stack:>9.4f} {pre_stack:>10.4f} {rec_stack:>7.4f} {f1_stack:>7.4f} {auc_stack:>7.4f}")
print("=" * 65)

# Pilih model terbaik
scores = {
    'Optimized SVM': auc_svm,
    'Optimized RF':  auc_rf,
    'Stacking':      auc_stack
}
best_model_name = max(scores, key=scores.get)
print(f"\n  [BEST] Model terbaik (berdasarkan AUC): {best_model_name} ({scores[best_model_name]:.4f})")

# ================= 13. Save Models =================
joblib.dump(best_svm_pipeline, 'model_svm_optimized.joblib')
joblib.dump(best_rf_pipeline,  'model_rf_optimized.joblib')
joblib.dump(stacking_model,    'model_stacking.joblib')

# Simpan threshold optimal
import json
thresholds_dict = {
    'svm_threshold': float(best_threshold_svm),
    'rf_threshold':  float(best_threshold_rf)
}
with open('optimal_thresholds.json', 'w') as f:
    json.dump(thresholds_dict, f, indent=2)

print("\n  [SAVED] Model tersimpan:")
print("    - model_svm_optimized.joblib")
print("    - model_rf_optimized.joblib")
print("    - model_stacking.joblib")
print("    - optimal_thresholds.json")
print("\n  Selesai! 100%")
