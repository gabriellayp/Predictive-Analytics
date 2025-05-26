# Proyek Klasifikasi Tingkat Risiko Mobil

## 🌍 Domain Proyek

Industri otomotif mengalami perkembangan pesat seiring meningkatnya permintaan konsumen terhadap mobil dengan berbagai spesifikasi. Namun, tidak semua mobil memiliki tingkat risiko yang sama, baik dari sisi keamanan, performa, maupun aspek ekonomi. Klasifikasi tingkat risiko mobil menjadi penting terutama bagi asuransi, dealer mobil, dan konsumen dalam mengambil keputusan.

Masalah ini harus diselesaikan karena informasi mengenai tingkat risiko dapat:

* Membantu perusahaan asuransi menetapkan premi yang lebih adil
* Membantu konsumen dalam memilih mobil sesuai kebutuhan dan preferensi risiko
* Membantu produsen memperbaiki fitur keselamatan dan performa berdasarkan umpan balik risiko

Referensi: Dataset yang digunakan merupakan dataset publik bertajuk "Automobile Data Set" yang tersedia di [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Automobile).

## 🔧 Business Understanding

### Problem Statement

Bagaimana memprediksi tingkat risiko sebuah mobil berdasarkan spesifikasi teknis dan fitur kategorikal?

### Goals

Membangun model klasifikasi untuk memprediksi tingkat risiko mobil: `low risk`, `medium risk`, atau `high risk`.

### Solution Statement

1. Membangun model baseline menggunakan algoritma Random Forest.
2. Melakukan eksperimen dengan algoritma lain: Gradient Boosting, XGBoost, dan AdaBoost.
3. Memilih model terbaik berdasarkan evaluasi metrik akurasi dan F1-score.

## 📊 Data Understanding

Dataset memiliki total **205 baris** data dan **26 fitur**, termasuk fitur numerik dan kategorikal.

Link dataset: [https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data)

### Fitur-Fitur Dataset:

* Fitur Numerik: `wheel-base`, `length`, `width`, `height`, `curb-weight`, `engine-size`, `horsepower`, `peak-rpm`, `price`, dll
* Fitur Kategorikal: `fuel-type`, `aspiration`, `body-style`, `drive-wheels`, `engine-location`, dll
* Target: `risk_level` (dihitung berdasarkan harga dan horsepower)

### Visualisasi Data

#### Distribusi Target

![Distribusi Risiko](./images/distribusi_risk_level.png)

#### Distribusi Fitur Kategorikal

![Distribusi Fitur Kategorikal 1](./images/distribusi_fitur_kategorikal_1.png)
![Distribusi Fitur Kategorikal 2](./images/distribusi_fitur_kategorikal_2.png)

#### Boxplot Fitur Numerik

![Boxplot Fitur Numerik](./images/boxplot_fitur_numerik.png)

## ⚙️ Data Preparation

* Menghapus missing value (seperti pada `normalized-losses` dan `num-of-doors`)
* Encoding fitur kategorikal menggunakan One-Hot Encoding
* Scaling fitur numerik menggunakan StandardScaler
* Menangani outlier dengan teknik IQR untuk fitur numerik utama

**Alasan:**

* Missing value dapat mengganggu model
* Kategori perlu direpresentasikan numerik agar bisa diproses oleh model
* Scaling diperlukan untuk model seperti XGBoost dan Gradient Boosting

## 📊 Modeling

### Algoritma yang Digunakan

1. **Random Forest**: Baseline model, cukup kuat dan menangani overfitting
2. **Gradient Boosting**: Lebih fokus pada kesalahan model sebelumnya
3. **XGBoost**: Optimasi lanjutan dari boosting yang efisien
4. **AdaBoost**: Fokus pada kesalahan prediksi yang sulit diklasifikasikan

### Parameter yang Digunakan

* RandomForest: `n_estimators=100`
* XGBoost: `max_depth=5`, `learning_rate=0.1`
* GradientBoosting: default
* AdaBoost: `n_estimators=100`

### Kelebihan & Kekurangan

| Algoritma         | Kelebihan               | Kekurangan                    |
| ----------------- | ----------------------- | ----------------------------- |
| Random Forest     | Stabil, cepat           | Bisa overfitting tanpa tuning |
| XGBoost           | Akurasi tinggi, efisien | Parameter banyak, kompleks    |
| Gradient Boosting | Baik untuk data imbang  | Cenderung lambat              |
| AdaBoost          | Fokus pada kesalahan    | Rentan outlier                |

### Model Terbaik

**XGBoost** memberikan akurasi terbaik: **92.31%**, dengan F1-score rata-rata **0.92**.

## 🎯 Evaluation

### Metrik Evaluasi

* **Accuracy**: proporsi prediksi yang benar
* **F1-score**: harmonic mean dari precision dan recall

### Hasil Evaluasi

| Model             | Accuracy   | Macro F1 |
| ----------------- | ---------- | -------- |
| Random Forest     | 89.23%     | 0.89     |
| Gradient Boosting | 90.77%     | 0.91     |
| AdaBoost          | 84.61%     | 0.84     |
| XGBoost           | **92.31%** | **0.92** |

## 📄 Struktur Laporan

Laporan ini disusun dengan format markdown untuk kompatibilitas dengan GitHub.

* Setiap bagian dilengkapi dengan visualisasi dan penjelasan proses
* Diagram dan gambar dimuat dalam folder `images/`
* Kode utama tersedia dalam notebook `classification_pipeline.ipynb`

## 📆 Resources

* Dataset: [https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data)
* Dokumentasi XGBoost: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
* Dokumentasi Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)

---

**🚀 Kesimpulan:** Model klasifikasi tingkat risiko mobil berhasil dibangun dengan performa terbaik menggunakan algoritma XGBoost. Model ini dapat digunakan untuk membantu perusahaan asuransi dan konsumen memahami risiko relatif dari suatu kendaraan berdasarkan spesifikasi teknis.
