import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Encoding kategorikal — OrdinalEncoder lebih stabil dari LabelEncoder untuk DataFrame
cat_cols = df.select_dtypes(include='object').columns.tolist()
df[cat_cols] = OrdinalEncoder().fit_transform(df[cat_cols])

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

""" ================= Tuning SVM =================  """
# RandomizedSearchCV lebih efisien dari GridSearchCV untuk ruang parameter besar
# loguniform memungkinkan eksplorasi C & gamma dalam rentang logaritmik yang luas

param_dist = {
    'C'     : loguniform(0.01, 1000),   # cari di rentang luas [0.01, 1000]
    'gamma' : loguniform(0.001, 10),
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_distributions=param_dist,
    n_iter=100,          # coba 100 kombinasi acak
    cv=10,               # 10-fold untuk estimasi lebih stabil
    scoring='roc_auc',   # AUC lebih informatif dari accuracy untuk data medis
    n_jobs=-1,
    random_state=42
)
search.fit(X_train_sc, y_train)
best = search.best_estimator_

print(f"\nBest params : {search.best_params_}")
print(f"Best CV AUC : {search.best_score_:.4f}")

""" ================= Evaluasi =================  """

y_pred = best.predict(X_test_sc)
y_prob = best.predict_proba(X_test_sc)[:, 1]

auc = roc_auc_score(y_test, y_prob)
cv  = cross_val_score(best, X_train_sc, y_train, cv=10, scoring='roc_auc')

print(f"\nTest AUC      : {auc:.4f}")
print(f"10-Fold CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Heart Disease'])}")

""" ================= Akurasi =================  """

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("=" * 40)
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print("=" * 40)

""" ================= Visualisasi =================  """

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Heart Disease'],
            yticklabels=['Normal', 'Heart Disease'])
axes[0].set(title=f'Confusion Matrix (AUC={auc:.3f})',
            ylabel='Actual', xlabel='Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'SVM (AUC={auc:.3f})')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.08, color='steelblue')
axes[1].set(title='ROC Curve — SVM',
            xlabel='False Positive Rate', ylabel='True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig('svm_result.png', dpi=150, bbox_inches='tight')
plt.show()