# 🩺 Sleep Disorder Prediction System (Hybrid ML & Expert System)

Aplikasi web berbasis Flask untuk memprediksi gangguan tidur (Normal, Insomnia, Sleep Apnea) menggunakan algoritma *Machine Learning* yang diperkuat dengan *Clinical Expert System Override*. Proyek ini dikembangkan sebagai bagian dari penelitian tugas akhir (Skripsi) di UIN Ar-Raniry Banda Aceh.

![Project Status](https://img.shields.io/badge/Status-Production_Ready-success)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-black)
![ML](https://img.shields.io/badge/Machine_Learning-XGBoost%20%7C%20Random_Forest-orange)

## ✨ Fitur Utama

Sistem ini tidak sekadar menebak berdasarkan data, melainkan menerapkan arsitektur *safeguard* berlapis layaknya sistem teknologi kesehatan (*HealthTech*) komersial:

1. **Bias-Free "Robust" Models:** Menghapus fitur demografis (`Gender` dan `Occupation`) dari dataset asli untuk mencegah model menghafal korelasi semu (*spurious correlation*), memaksa algoritma untuk murni belajar dari indikator klinis dan gaya hidup.
2. **Interactive Model Comparison:** Dilengkapi dengan fitur *dropdown* di antarmuka web untuk membandingkan 3 model secara *real-time* tanpa *reload* server:
   - **XGBoost** (Model Utama)
   - **Random Forest** (Tuned Model)
   - **Logistic Regression** (Baseline Model)
3. **Clinical Expert System Override:** *Rule-based engine* yang membajak prediksi *Machine Learning* apabila input pengguna menunjukkan kondisi medis ekstrem yang berbahaya (misal: kurang tidur ekstrem + stres tinggi, atau Obesitas + Hipertensi). Ini memastikan prinsip *Zero False Negative* untuk keselamatan pengguna.
4. **Explainable AI (Health Insights):** Antarmuka cerdas yang tidak hanya memberikan hasil prediksi, tetapi juga memecah metrik pengguna menjadi daftar "✅ Kebiasaan Baik" dan "⚠️ Perlu Ditingkatkan" berdasarkan standar medis (WHO/AHA).
5. **Strict OOD Validation:** Validasi *Out-of-Distribution* di sisi *backend* untuk menolak input yang tidak masuk akal (misalnya detak jantung 900 bpm).

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** XGBoost, Scikit-Learn (Random Forest, Logistic Regression, SMOTE)
- **Data Manipulation:** Pandas, NumPy
- **Frontend:** HTML5, Jinja2, Bootstrap (Responsive UI)


## Quick Start

Ikuti langkah-langkah berikut untuk menjalankan proyek ini secara lokal:

**1. Clone Repositori**
```bash
git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
cd nama-repo-kamu
