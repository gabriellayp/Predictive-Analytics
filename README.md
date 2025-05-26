# Laporan Proyek Machine Learning – [Nama Anda]

---

## 1. Domain Proyek

Industri otomotif dan asuransi menghadapi tantangan dalam menentukan tingkat risiko kendaraan secara cepat dan akurat. Penilaian risiko mobil penting untuk menentukan premi asuransi dan keputusan bisnis lainnya. Data teknis mobil seperti ukuran mesin, berat, dan jenis bahan bakar bisa memberikan sinyal penting terkait risiko kecelakaan atau kerugian finansial. Oleh karena itu, proyek ini bertujuan membangun model machine learning yang dapat mengklasifikasikan tingkat risiko mobil secara otomatis berdasarkan data historis dari 1985 Ward’s Automotive Yearbook.

Menurut McKinsey & Company (2020), digitalisasi dan analitik prediktif dalam asuransi meningkatkan ketepatan underwriting dan pengelolaan risiko, sehingga perusahaan bisa mengurangi kerugian dan meningkatkan profitabilitas. Dengan model yang efektif, proses penentuan risiko menjadi lebih efisien dan berbasis data.

**Referensi:**  
McKinsey & Company, “Insurance 2030—The impact of AI on the future of insurance,” 2020.  
[https://www.mckinsey.com/industries/financial-services/our-insights/insurance-2030-the-impact-of-ai-on-the-future-of-insurance](https://www.mckinsey.com/industries/financial-services/our-insights/insurance-2030-the-impact-of-ai-on-the-future-of-insurance)

---

## 2. Business Understanding

### Problem Statements

1. Bagaimana mengklasifikasikan risiko kendaraan berdasarkan atribut teknis seperti bobot, tipe mesin, dan ukuran kendaraan?  
2. Apakah model machine learning dapat memberikan prediksi yang akurat untuk tingkat risiko mobil guna mendukung keputusan bisnis?

### Goals

1. Membangun model klasifikasi tingkat risiko kendaraan dengan kategori *low risk*, *medium risk*, dan *high risk*.  
2. Mengevaluasi performa beberapa algoritma klasifikasi dan memilih model terbaik berdasarkan metrik evaluasi.

### Solution Statements

- Menggunakan minimal dua algoritma klasifikasi: Gradient Boosting Classifier dan XGBoost Classifier.  
- Melakukan hyperparameter tuning untuk meningkatkan performa model.  
- Evaluasi model dilakukan menggunakan metrik akurasi, precision, recall, dan F1-score.  
- Model terbaik dipilih berdasarkan metrik evaluasi tersebut untuk implementasi akhir.

---

## 3. Data Understanding

Dataset yang digunakan berasal dari UCI Machine Learning Repository: [Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/Automobile). Dataset ini mengandung 205 sampel mobil dengan 26 fitur, meliputi spesifikasi teknis, harga, dan rating risiko.

### Informasi Data:

- **Jumlah data:** 205 baris (mobil)  
- **Jumlah fitur:** 26 kolom  
- **Target:** *risk_level* yang dibuat dari kolom `symboling`  
- **Kondisi data:** Terdapat nilai hilang pada beberapa fitur seperti `normalized-losses`, `price`, dan lain-lain.

### Fitur-fitur Utama:

| Fitur             | Deskripsi                                                   |
|-------------------|-------------------------------------------------------------|
| symboling         | Risiko mobil (-3 = aman, +3 = berisiko tinggi)              |
| normalized-losses | Rata-rata kerugian dinormalisasi                            |
| make              | Merek mobil                                                |
| fuel-type         | Jenis bahan bakar (gas/petrol)                             |
| aspiration        | Sistem pembakaran (normal/turbo)                           |
| num-of-doors      | Jumlah pintu                                               |
| body-style        | Bentuk bodi kendaraan                                      |
| drive-wheels      | Sistem penggerak roda                                      |
| engine-location   | Lokasi mesin                                              |
| wheel-base        | Jarak sumbu roda                                          |
| length, width, height | Dimensi mobil                                            |
| curb-weight       | Berat mobil                                               |
| engine-type       | Tipe mesin                                               |
| num-of-cylinders  | Jumlah silinder                                          |
| engine-size       | Ukuran mesin                                           |
| fuel-system       | Sistem bahan bakar                                     |
| bore, stroke      | Ukuran piston                                          |
| compression-ratio | Rasio kompresi                                        |
| horsepower       | Daya mesin                                             |
| peak-rpm          | Putaran maksimum mesin                                 |
| city-mpg, highway-mpg | Efisiensi bahan bakar perkotaan dan antar kota      |
| price             | Harga kendaraan                                      |

### Exploratory Data Analysis (EDA):

- Distribusi target `symboling` menunjukkan kelas tidak seimbang, sehingga dilakukan grouping menjadi tiga kelas risiko.  
- Beberapa fitur numerik menunjukkan korelasi signifikan terhadap risiko, seperti `curb-weight` dan `horsepower`.  
- Nilai hilang ditangani agar tidak mengganggu proses modeling.

---

## 4. Data Preparation

### Langkah yang dilakukan:

1. **Penanganan nilai hilang**  
   - Numerik: diimputasi menggunakan median untuk menjaga kestabilan distribusi.  
   - Kategorik: diimputasi menggunakan modus agar nilai dominan tidak hilang.  
   
2. **Encoding fitur kategorik**  
   - Menggunakan One-Hot Encoding agar fitur dapat diproses model berbasis numerik.  

3. **Pembuatan label target**  
   - Kolom `symboling` dikonversi menjadi tiga kelas risiko:  
     - Low Risk: symboling -3, -2, -1  
     - Medium Risk: symboling 0  
     - High Risk: symboling 1, 2, 3  

4. **Normalisasi fitur numerik**  
   - Menggunakan StandardScaler untuk skala fitur agar semua fitur numerik berada pada skala yang sama dan mempercepat konvergensi model.  

5. **Split data**  
   - Data dibagi 80% untuk training, 20% untuk testing.

### Alasan teknik data preparation:

- Imputasi diperlukan untuk mengatasi missing values yang dapat menyebabkan error dan bias model.  
- Encoding diperlukan agar data kategorik dapat digunakan oleh algoritma machine learning yang membutuhkan input numerik.  
- Normalisasi membuat proses pelatihan lebih stabil dan efisien.  
- Split data agar evaluasi model pada data tidak pernah dilihat sebelumnya (unseen data).

---

## 5. Modeling

### Algoritma yang digunakan:

1. **Gradient Boosting Classifier (GBC)**  
   - Parameter utama: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.  
   - Kelebihan: kuat terhadap overfitting, efektif untuk data tabular.  
   - Kekurangan: lambat pada dataset besar.

2. **XGBoost Classifier**  
   - Parameter utama disesuaikan dengan `GridSearchCV`: `n_estimators`, `max_depth`, `learning_rate`.  
   - Kelebihan: efisien, mendukung regularisasi, cocok untuk dataset tabular.  
   - Kekurangan: tuning parameter lebih kompleks.

3. **AdaBoost Classifier**  
   - Base estimator: DecisionTree stump (max_depth=1).  
   - Parameter utama: `n_estimators=50`, `learning_rate=1.0`.  
   - Kelebihan: mudah dipahami, efektif untuk dataset kecil.  
   - Kekurangan: sensitivitas terhadap noisy data.

### Proses Improvement:

- Melakukan hyperparameter tuning menggunakan `GridSearchCV` untuk memilih kombinasi parameter terbaik.  
- Contoh snippet tuning XGBoost:

```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
