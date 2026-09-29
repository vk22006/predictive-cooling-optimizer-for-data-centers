# Pengoptimalan Pendinginan Prediktif untuk Pusat Data: Penjadwalan Chiller Berbasis Suhu untuk Mengurangi Konsumsi Energi
[English](README.md) | [தமிழ்](README_TA.md) | [中文](README_ZH.md) | [हिन्दी](README_HI.md) | Bahasa Indonesia

![GitHub top language](https://img.shields.io/github/languages/top/vk22006/predictive-cooling-optimizer-for-data-centers)
![GitHub language count](https://img.shields.io/github/languages/count/vk22006/predictive-cooling-optimizer-for-data-centers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![GitHub forks](https://img.shields.io/github/forks/vk22006/predictive-cooling-optimizer-for-data-centers)

Proyek ini bertujuan untuk mengatasi ketidakefisienan energi pada sistem pendinginan pusat data dengan mengembangkan model prediktif berbasis suhu yang mengoptimalkan penjadwalan chiller untuk mengurangi konsumsi energi sambil mempertahankan keamanan termal. Sistem pendinginan reaktif tradisional merespons perubahan suhu setelah perubahan tersebut terjadi, sehingga menyebabkan pemborosan energi dan pengoperasian chiller yang kurang optimal.

![Halaman utama](img/home_page.PNG "Halaman utama")

## Metodologi Proyek

Metodologi dimulai dengan prapemrosesan data secara menyeluruh terhadap 13.615 sampel HVAC, termasuk deteksi outlier menggunakan IQR, normalisasi melalui MinMaxScaler, serta pembagian data pelatihan dan pengujian secara kronologis dengan rasio 80-20 untuk menjaga integritas temporal data.

Feature engineering menghasilkan 46 fitur tambahan yang mencakup 16 lag features, 12 rolling averages, 6 cyclical temporal encodings, dan 4 interaction features. Fitur-fitur tersebut digunakan untuk menangkap dinamika sistem yang kompleks dengan lebih baik.

Dua model regresi XGBoost menjadi inti mesin prediksi:

* **Energy Prediction Model** mencapai R² = 0.9891 dengan MAE sebesar 1.222 kWh.
* **Temperature Forecasting Model** mencapai R² = 0.6853, dengan 89.24% prediksi berada dalam toleransi ±1°C.

Kedua model menunjukkan waktu pelatihan yang efisien. Energy Prediction Model dilatih dalam 2,12 detik, sedangkan Temperature Forecasting Model membutuhkan 1,87 detik. Hal ini menunjukkan bahwa kedua model memiliki potensi untuk digunakan dalam penerapan secara real-time.

Class `PredictiveCoolingOptimizer` mengintegrasikan kedua model tersebut dan memungkinkan optimasi sistem secara menyeluruh melalui strategi manajemen suhu berbasis batasan (constraint-based) serta strategi minimisasi konsumsi energi.

## Pengujian

Sebanyak 11 pengujian dilakukan dalam lima kategori. Berikut rinciannya:

|     Pengujian     |                   Target                   |   Status   |
| :---------------: | :----------------------------------------: | :--------: |
|     Unit Tests    | Model energi dan suhu, Optimization Engine | ✅ Berhasil |
| Integration Tests |   End-to-End Pipeline, System Integration  | ✅ Berhasil |
|  Functional Tests |     Akurasi, waktu respons, dan logika     | ✅ Berhasil |
|   White Box Test  |    Hyperparameters, Feature Engineering    | ✅ Berhasil |
|   Black Box Test  |     Boundary Values, Output Consistency    | ✅ Berhasil |
|                   |             Pengujian berhasil             |    11/11   |
|                   |               Pengujian gagal              |    0/11    |
|                   |            Tingkat keberhasilan            |   100.0%   |

## Prosedur Menjalankan Program

Proses menjalankan program cukup sederhana. Ikuti langkah-langkah berikut.

1. Instal library yang diperlukan:

```bash
pip install xgboost streamlit
```

2. Buka Command Prompt atau PowerShell dan masuk ke folder proyek:

```bash
cd <your-file-path>
```

3. Jalankan aplikasi menggunakan perintah berikut:

```bash
streamlit run 1_Home.py
```

## Tools yang Digunakan

1. Anaconda Jupyter - untuk pelatihan dan pengujian model
2. Streamlit Library - untuk implementasi frontend
3. Joblib - untuk menangani file model `.pkl`
4. NumPy
5. Pandas
6. Scikit-Learn
7. XGBoost

## Algoritma yang Digunakan

### 1. Algoritma Prediksi

* XGBoost (Extreme Gradient Boosting)
* Random Forest Regressor

### 2. Algoritma Pendukung

* Min-Max Normalization
* Rolling Average (untuk feature engineering)

Tool ini berhasil menunjukkan kelayakan optimasi pendinginan prediktif berbasis perangkat lunak untuk pusat data. Model yang telah dilatih siap untuk diterapkan pada aplikasi web interaktif menggunakan Streamlit, sehingga mendukung antarmuka yang mudah digunakan serta demonstrasi sistem kepada pengguna dan pemangku kepentingan.

## Lisensi

Proyek ini dilisensikan di bawah MIT License. Lihat file [LICENSE](LICENSE) untuk informasi lebih lanjut.
