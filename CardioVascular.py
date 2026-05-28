import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

# ================= 3. Split Fitur & Target =================
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)

# ================= 4. Konfigurasi Pipeline =================
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numeric_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

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
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
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