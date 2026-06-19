# 💗 Sistem Prediksi Penyakit Kardiovaskular

Aplikasi Machine Learning berbasis **SVM (Support Vector Machine)** dan **Random Forest** untuk memprediksi risiko penyakit jantung berdasarkan data rekam medis pasien.

---

## 📁 Struktur Folder

```
Machine-Learning-SVM/
│
├── CardioVascular.py               # Script utama: training, evaluasi, & simpan model
├── app.py                          # Aplikasi web Streamlit untuk prediksi
├── heart.csv                       # Dataset (Heart Failure Prediction - Kaggle)
├── Requirements.txt                # Daftar library yang dibutuhkan
│
├── Save_model/                     # Model & artefak hasil training
│   ├── best_model.pkl              # Model terbaik berdasarkan ROC-AUC
│   ├── tuned_svm.pkl               # Tuned SVM (untuk pilihan di app)
│   ├── tuned_rf.pkl                # Tuned Random Forest (untuk pilihan di app)
│   ├── preprocessor.pkl            # ColumnTransformer (OHE + passthrough)
│   ├── scaler.pkl                  # StandardScaler (khusus SVM)
│   ├── clip_bounds.pkl             # Batas winsorize dari data training
│   ├── feature_names.pkl           # Nama fitur setelah encoding
│   └── best_label.pkl              # Label model terbaik
│
├── baseline_confusion_matrix.png   # Visualisasi confusion matrix baseline
├── tuned_confusion_matrix.png      # Visualisasi confusion matrix tuned
├── roc_curve_all.png               # ROC Curve semua model
└── comparison_metrics.png          # Bar chart perbandingan metrik
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r Requirements.txt
```

### 2. Training Model
Jalankan script training untuk melatih model dan menyimpan hasilnya:
```bash
python CardioVascular.py
```
Script ini akan:
- Memuat dan membersihkan dataset `heart.csv`
- Melatih Baseline SVM & Random Forest
- Melakukan hyperparameter tuning dengan RandomizedSearchCV
- Menyimpan model ke folder `Save_model/`
- Menghasilkan visualisasi evaluasi (PNG)

### 3. Jalankan Aplikasi Web
```bash
streamlit run app.py
```
Buka browser di **http://localhost:8501**

---

## 📊 Hasil Evaluasi Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline SVM | 0.8533 | 0.8505 | 0.8922 | 0.8708 | 0.9412 |
| **Tuned SVM** | **0.8750** | **0.8835** | **0.8922** | **0.8878** | **0.9359** |
| Baseline RF | 0.8696 | 0.8824 | 0.8824 | 0.8824 | 0.9296 |
| Tuned RF | 0.8696 | 0.8900 | 0.8725 | 0.8812 | 0.9274 |

> Model terbaik dipilih berdasarkan nilai **ROC-AUC** tertinggi.

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Nama** | Heart Failure Prediction Dataset |
| **Sumber** | [Kaggle — fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| **Jumlah Data** | 918 pasien |
| **Fitur** | 11 fitur klinis |
| **Target** | `HeartDisease` (0 = Normal, 1 = Heart Disease) |
| **Split** | 80% Training / 20% Testing |

### Fitur Input

| Fitur | Tipe | Keterangan |
|---|---|---|
| Age | Numerik | Usia pasien (tahun) |
| Sex | Kategorikal | Jenis kelamin (M/F) |
| ChestPainType | Kategorikal | Tipe nyeri dada (ATA/NAP/ASY/TA) |
| RestingBP | Numerik | Tekanan darah istirahat (mmHg) |
| Cholesterol | Numerik | Kolesterol serum (mg/dL) |
| FastingBS | Biner | Gula darah puasa > 120 mg/dL (0/1) |
| RestingECG | Kategorikal | Hasil ECG istirahat (Normal/ST/LVH) |
| MaxHR | Numerik | Detak jantung maksimum |
| ExerciseAngina | Kategorikal | Angina saat olahraga (Y/N) |
| Oldpeak | Numerik | ST depression akibat olahraga |
| ST_Slope | Kategorikal | Kemiringan segmen ST (Up/Flat/Down) |

---

## 🤖 Informasi Teknis

| Komponen | Detail |
|---|---|
| **Algoritma** | SVM & Random Forest |
| **Tuning** | RandomizedSearchCV |
| **Cross-Validation** | 5-Fold CV |
| **Scoring Tuning** | ROC-AUC |
| **Preprocessing** | Winsorize + OneHotEncoding + StandardScaler |
| **Framework ML** | Scikit-learn |
| **UI** | Streamlit |
| **Bahasa** | Python 3 |

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

> Install semua dengan: `pip install -r Requirements.txt`

---

## 👥 Tim Pengembang

Proyek Tugas Besar Machine Learning — Semester 4  
Institut Teknologi Kalimantan (ITK)
