# Laporan Proyek Machine Learning – \[Gabriella Yoanda Pelawi]

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

1. Bagaimana cara mengklasifikasikan tingkat risiko kendaraan berdasarkan fitur-fitur teknis mobil?
2. Apakah model machine learning dapat memberikan prediksi yang akurat untuk tingkat risiko mobil guna mendukung keputusan bisnis?

### Goals

1. Membangun model klasifikasi tingkat risiko kendaraan dengan kategori *low risk*, *medium risk*, dan *high risk*.
2. Mengevaluasi performa beberapa algoritma klasifikasi dan memilih model terbaik berdasarkan metrik evaluasi.
3. Menghasilkan insight yang relevan bagi pengambilan keputusan di sektor otomotif dan asuransi.

### Solution Statements

* Menggunakan minimal tiga algoritma klasifikasi: Gradient Boosting Classifier, XGBoost Classifier dan AdaBoost.
* Melakukan hyperparameter tuning untuk meningkatkan performa model.
* Evaluasi model dilakukan menggunakan metrik akurasi, precision, recall, dan F1-score.
* Model terbaik dipilih berdasarkan metrik evaluasi tersebut untuk implementasi akhir.

---

## 3. Data Understanding

Dataset yang digunakan berasal dari UCI Machine Learning Repository: [Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/Automobile). Dataset ini mengandung 205 sampel mobil dengan 26 fitur, meliputi spesifikasi teknis, harga, dan rating risiko.

### Informasi Data:

* **Jumlah data:** 205 baris (mobil)
* **Jumlah fitur:** 26 kolom
* **Target:** *risk\_level* yang dibuat dari kolom `symboling`
* **Kondisi data:** Terdapat nilai hilang pada beberapa fitur seperti `normalized-losses`, `price`, dan lain-lain.

### Fitur-fitur Utama:

| Fitur                 | Deskripsi                                      |
| --------------------- | ---------------------------------------------- |
| symboling             | Risiko mobil (-3 = aman, +3 = berisiko tinggi) |
| normalized-losses     | Rata-rata kerugian dinormalisasi               |
| make                  | Merek mobil                                    |
| fuel-type             | Jenis bahan bakar (gas/petrol)                 |
| aspiration            | Sistem pembakaran (normal/turbo)               |
| num-of-doors          | Jumlah pintu                                   |
| body-style            | Bentuk bodi kendaraan                          |
| drive-wheels          | Sistem penggerak roda                          |
| engine-location       | Lokasi mesin                                   |
| wheel-base            | Jarak sumbu roda                               |
| length, width, height | Dimensi mobil                                  |
| curb-weight           | Berat mobil                                    |
| engine-type           | Tipe mesin                                     |
| num-of-cylinders      | Jumlah silinder                                |
| engine-size           | Ukuran mesin                                   |
| fuel-system           | Sistem bahan bakar                             |
| bore, stroke          | Ukuran piston                                  |
| compression-ratio     | Rasio kompresi                                 |
| horsepower            | Daya mesin                                     |
| peak-rpm              | Putaran maksimum mesin                         |
| city-mpg, highway-mpg | Efisiensi bahan bakar perkotaan dan antar kota |
| price                 | Harga kendaraan                                |

### Exploratory Data Analysis (EDA):
Untuk memahami karakteristik data yang digunakan, dilakukan eksplorasi data (EDA) terhadap variabel target dan fitur-fitur yang tersedia. Berikut ini merupakan hasil dari proses eksplorasi data:

#### Distribusi Target: risk\_level

Distribusi tingkat risiko mobil (risk\_level) menunjukkan bahwa mayoritas mobil berada pada kategori *low risk*, disusul oleh *high risk* dan *medium risk*. Ini menunjukkan bahwa dataset agak tidak seimbang, namun ketimpangannya masih bisa ditangani oleh model klasifikasi dengan pendekatan tertentu.

![Distribusi risk\_level](attachment\:image1.png)

#### Distribusi Fitur Kategorikal

Beberapa fitur kategorikal yang dianalisis antara lain:

* **Aspiration:** Mayoritas mobil memiliki konfigurasi *std* dibandingkan *turbo*.
* **Number of Doors:** Terdapat lebih banyak mobil dengan empat pintu dibandingkan dua pintu.
* **Fuel Type:** Bahan bakar *gas* lebih dominan daripada *diesel*.
* **Engine Location:** Mayoritas mobil memiliki mesin di bagian depan.

![Distribusi Fitur Kategorikal](attachment\:image2.png)

Fitur lainnya yang juga dieksplorasi:

* **Body Style:** Mobil dengan gaya bodi *sedan* mendominasi, diikuti oleh *hatchback*.
* **Drive Wheels:** Konfigurasi penggerak roda *fwd* (front-wheel drive) paling banyak.
* **Engine Type:** Tipe mesin *ohc* paling sering ditemukan.
* **Fuel System:** Sistem bahan bakar *mpfi* paling umum.

![Distribusi Variabel Kategorikal Lanjutan](attachment\:image3.png)

#### Distribusi Fitur Numerik

Boxplot digunakan untuk menganalisis sebaran dan outlier pada fitur numerik. Beberapa insight yang ditemukan:

* Fitur seperti *normalized-losses*, *compression-ratio*, *horsepower*, dan *price* memiliki banyak outlier.
* Fitur seperti *wheel-base*, *length*, *curb-weight*, dan *engine-size* memiliki distribusi yang relatif normal.
* Rentang nilai cukup bervariasi di hampir semua fitur numerik, menandakan perlunya normalisasi.

![Distribusi Fitur Numerik](attachment\:image4.png)

Hasil eksplorasi ini menjadi dasar dalam tahap preprocessing selanjutnya, termasuk penanganan missing value, encoding fitur kategorikal, dan scaling fitur numerik.


* Distribusi target `symboling` menunjukkan kelas tidak seimbang, sehingga dilakukan grouping menjadi tiga kelas risiko.
* Beberapa fitur numerik menunjukkan korelasi signifikan terhadap risiko, seperti `curb-weight` dan `horsepower`.
* Nilai hilang ditangani agar tidak mengganggu proses modeling.

---

## 4. Data Preparation

### Langkah yang dilakukan:

1. **Penanganan nilai hilang**

   * Numerik: diimputasi menggunakan median untuk menjaga kestabilan distribusi.
   * Kategorik: diimputasi menggunakan modus agar nilai dominan tidak hilang.

2. **Encoding fitur kategorik**

   * Menggunakan One-Hot Encoding agar fitur dapat diproses model berbasis numerik.

3. **Pembuatan label target**

   * Kolom `symboling` dikonversi menjadi tiga kelas risiko:

     * Low Risk: symboling -3, -2, -1
     * Medium Risk: symboling 0
     * High Risk: symboling 1, 2, 3

4. **Normalisasi fitur numerik**

   * Menggunakan StandardScaler untuk skala fitur agar semua fitur numerik berada pada skala yang sama dan mempercepat konvergensi model.

5. **Split data**

   * Data dibagi 80% untuk training, 20% untuk testing.

### Alasan teknik data preparation:

* Imputasi diperlukan untuk mengatasi missing values yang dapat menyebabkan error dan bias model.
* Encoding diperlukan agar data kategorik dapat digunakan oleh algoritma machine learning yang membutuhkan input numerik.
* Normalisasi membuat proses pelatihan lebih stabil dan efisien.
* Split data agar evaluasi model pada data tidak pernah dilihat sebelumnya (unseen data).

---

## 5. Modeling

### Algoritma yang digunakan:
Proyek ini menggunakan empat algoritma klasifikasi yang umum digunakan untuk data tabular:

1. **Random Forest Classifier**
   - Algoritma ensemble berbasis pohon keputusan.
   - Parameter default: `n_estimators=100`.
   - Kelebihan: tahan terhadap overfitting, stabil, dan bekerja baik dengan data kategorikal maupun numerik.
   - Kekurangan: bisa kurang efisien pada dataset yang sangat besar.

2. **Gradient Boosting Classifier (GBC)**
   - Teknik boosting yang membangun model secara bertahap.
   - Parameter utama: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`.
   - Kelebihan: kuat terhadap overfitting, efektif untuk data tabular.
   - Kekurangan: lambat pada dataset besar.

3. **XGBoost Classifier**
   - Versi optimasi dari boosting dengan regularisasi.
   - Parameter terbaik hasil `GridSearchCV`:  
     `n_estimators=150`, `max_depth=5`, `learning_rate=0.1`.
   - Kelebihan: cepat, efisien, mendukung regulasi, unggul di banyak kompetisi.
   - Kekurangan: tuning parameter lebih kompleks.

4. **AdaBoost Classifier**
   - Menggabungkan banyak decision stump (max_depth=1).
   - Parameter default: `n_estimators=50`, `learning_rate=1.0`.
   - Kelebihan: sederhana, efektif pada data bersih.
   - Kekurangan: sensitif terhadap outlier dan data noisy.

## D. Evaluation & Results

### 1. Metrik Evaluasi

- **Accuracy**: Persentase prediksi yang benar dari seluruh data.
- **Precision**: Ketepatan model dalam memprediksi kelas tertentu.
- **Recall**: Seberapa baik model menangkap seluruh data aktual pada suatu kelas.
- **F1-Score**: Rata-rata harmonik dari precision dan recall.

### 2. Hasil Evaluasi Model

| Model               | Accuracy   |
|---------------------|------------|
| Random Forest       | 88%        |
| Gradient Boosting   | 90.38%     |
| **XGBoost**         | **92.31%** |
| AdaBoost            | 86.53%     |

## E. Kesimpulan

- **Model XGBoost memberikan performa terbaik** dengan akurasi 92.31%, mengungguli Gradient Boosting (90.38%), Random Forest (88%), dan AdaBoost (86.53%).
- Fitur-fitur seperti `curb-weight`, `engine-size`, `horsepower`, dan `price` berkontribusi besar dalam penentuan risiko kendaraan.
- Algoritma boosting (GBC, XGBoost, AdaBoost) menunjukkan performa lebih unggul dibandingkan metode ensemble lainnya (Random Forest) pada dataset ini.
- Pemilihan model berbasis boosting sangat direkomendasikan untuk kasus klasifikasi risiko otomotif berbasis fitur teknis.

---

## F. Saran dan Pekerjaan Selanjutnya

- Melakukan validasi silang menggunakan metode seperti **Stratified K-Fold** untuk evaluasi model yang lebih robust.
- Mengeksplorasi teknik **feature engineering lanjutan**, seperti interaksi antar fitur atau fitur turunan.
- Membangun pipeline produksi dan menyimpan model dalam format `.pkl` atau `.joblib` untuk keperluan deployment API atau integrasi dengan dashboard.
- Menguji model pada dataset otomotif terbaru untuk menguji generalisasi dan daya adaptasi model terhadap tren teknologi kendaraan modern.

---

## G. Resources

- Dataset: [UCI Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/Automobile)
- Artikel Referensi:  
  McKinsey & Company – “Insurance 2030: The impact of AI on the future of insurance”  
  [https://www.mckinsey.com/industries/financial-services/our-insights/insurance-2030-the-impact-of-ai-on-the-future-of-insurance](https://www.mckinsey.com/industries/financial-services/our-insights/insurance-2030-the-impact-of-ai-on-the-future-of-insurance)

