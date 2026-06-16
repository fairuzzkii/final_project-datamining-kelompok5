# 📊 Analisis Pengaruh Kemiskinan dan Sanitasi terhadap Prevalensi Stunting di Indonesia

Aplikasi ini merupakan bagian dari tahapan **CRISP-DM Fase 6: Deployment** yang diwujudkan dalam bentuk **Streamlit Interactive Dashboard**. Aplikasi ini memodelkan dan memetakan hubungan antara tingkat ekonomi (kemiskinan) dan infrastruktur lingkungan (sanitasi layak) terhadap kejadian stunting di 34 provinsi di Indonesia (2021-2024).

---

## 🔬 Variabel Riset
Model analisis di dalam dashboard dan notebook ini didasarkan pada 3 variabel utama:
- **Prevalensi Balita Stunting ($Y$)** - Variabel Terikat (Kemenkes RI)
- **Persentase Penduduk Miskin ($X_1$)** - Variabel Bebas 1 (BPS RI)
- **Akses Sanitasi Layak ($X_2$)** - Variabel Bebas 2 (BPS RI)

---


## 🛠️ Instalasi & Persiapan
Sebelum menjalankan aplikasi, pastikan Anda telah menginstal seluruh library Python yang dibutuhkan. Buka terminal atau Command Prompt di folder utama proyek, lalu jalankan:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Menjalankan Dashboard Streamlit
Untuk membuka dashboard visualisasi interaktif di browser Anda, jalankan perintah berikut dari **root folder (folder utama datmin)**:

```bash
streamlit run app.py
```

### 2. Menjalankan Notebook CRISP-DM
Notebook di dalam folder `notebooks/` telah diatur menggunakan *path resolver* otomatis. Anda dapat menjalankannya dengan dua cara:
- **Secara Lokal**: Jalankan server Jupyter (`jupyter notebook`), buka folder `notebooks/`, lalu jalankan `crisp_dm_kemiskinan_stunting.ipynb`. Kode secara otomatis membaca data mentah dari folder `../dataset/`.
- **Google Colab**: Unggah file `crisp_dm_kemiskinan_stunting.ipynb` ke Google Colab. Unggah juga seluruh 9 file `.csv` dari folder `dataset/` ke direktori root Colab (`/content/`). Notebook akan otomatis mendeteksi file dan berjalan lancar tanpa error.

---

## 📐 Fitur Utama Dashboard
1. **Eksplorasi Data & Tren**: Menampilkan statistik deskriptif tingkat nasional, grafik korelasi Pearson, heatmap korelasi, dan visualisasi tren temporal (2021-2024).
2. **Sistem Prediksi Stunting**: Simulasi kalkulator prediksi stunting menggunakan model *Multiple Linear Regression* dengan input slider kemiskinan dan sanitasi secara *real-time*.
3. **Pemetaan Zona Risiko**: Pemetaan 3D sebaran spasial 34 provinsi Indonesia ke dalam 3 zona risiko (Tinggi, Menengah, Rendah) berbasis algoritma *K-Means Clustering*.
