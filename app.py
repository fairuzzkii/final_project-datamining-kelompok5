# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import pearsonr
import plotly.express as px
import pickle
import os
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Kemiskinan, Sanitasi & Stunting Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] .stRadio label {
    color: #e0e0e0 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
    border: 1px solid #667eea44;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
}

div[data-testid="stMetric"] label { 
    font-weight: 600; 
}

.main-header {
    font-size: 2.2rem; 
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
    margin-bottom: 0; 
    line-height: 1.2;
}

.sub-header { 
    font-size: 1.0rem; 
    color: #888; 
    margin-top: 0; 
    margin-bottom: 24px; 
}

.risk-card {
    border-radius: 14px; 
    padding: 22px 26px; 
    margin-bottom: 14px;
    color: white; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    transition: transform 0.2s;
}

.risk-card:hover { 
    transform: translateY(-3px); 
}

.risk-high { 
    background: linear-gradient(135deg, #e53935, #b71c1c); 
}

.risk-medium { 
    background: linear-gradient(135deg, #fb8c00, #ef6c00); 
}

.risk-low { 
    background: linear-gradient(135deg, #43a047, #2e7d32); 
}

.risk-card h3 { 
    margin: 0 0 8px; 
    font-size: 1.15rem; 
}

.risk-card p { 
    margin: 4px 0; 
    font-size: 0.9rem; 
    opacity: 0.92; 
}

.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; 
    border-radius: 16px; 
    padding: 30px; 
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); 
    margin: 20px 0;
}

.prediction-box h2 { 
    margin: 0; 
    font-size: 2.8rem; 
    font-weight: 800; 
}

.prediction-box p { 
    margin: 8px 0 0; 
    font-size: 1.0rem; 
    opacity: 0.9; 
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATH RESOLVER HELPER
# ─────────────────────────────────────────────────────────────────────────────
def resolve_path(filename, folder=""):
    if not folder:
        return filename
    return os.path.join(folder, filename)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER & MODEL HELPER
# ─────────────────────────────────────────────────────────────────────────────
VALID_34 = {
    'ACEH','SUMATERA UTARA','SUMATERA BARAT','RIAU','JAMBI','SUMATERA SELATAN',
    'BENGKULU','LAMPUNG','KEPULAUAN BANGKA BELITUNG','KEPULAUAN RIAU',
    'DKI JAKARTA','JAWA BARAT','JAWA TENGAH','DI YOGYAKARTA','JAWA TIMUR',
    'BANTEN','BALI','NUSA TENGGARA BARAT','NUSA TENGGARA TIMUR',
    'KALIMANTAN BARAT','KALIMANTAN TENGAH','KALIMANTAN SELATAN',
    'KALIMANTAN TIMUR','KALIMANTAN UTARA',
    'SULAWESI UTARA','SULAWESI TENGAH','SULAWESI SELATAN','SULAWESI TENGGARA',
    'GORONTALO','SULAWESI BARAT','MALUKU','MALUKU UTARA','PAPUA BARAT','PAPUA'
}

SANITASI_NAME_MAP = {
    'KEP. BANGKA BELITUNG': 'KEPULAUAN BANGKA BELITUNG',
    'KEP. RIAU': 'KEPULAUAN RIAU',
}

JUNK_ROWS = {'INDONESIA', '-', 'CATATAN', '',
             'PAPUA BARAT DAYA', 'PAPUA SELATAN', 'PAPUA TENGAH', 'PAPUA PEGUNUNGAN'}

@st.cache_data
def load_data():
    final_dataset_path = resolve_path("dataset_final_kemiskinan_stunting.csv", "dataset/processed")
    avg_cluster_path = resolve_path("dataset_avg_cluster.csv", "dataset/processed")

    if os.path.exists(final_dataset_path):
        df = pd.read_csv(final_dataset_path)
        df_avg = None
        if os.path.exists(avg_cluster_path):
            df_avg = pd.read_csv(avg_cluster_path)
        return df, df_avg

    stunting_raw_file = "vertikalkementerian-2-od_20953_prevalensi_balita_stunting_brdsrkn_prov_di_indones_v1_data.csv"
    stunting_file = resolve_path(stunting_raw_file, "dataset/raw")
    if not os.path.exists(stunting_file):
        st.error(f"❌ File tidak ditemukan: `{stunting_file}`")
        st.stop()
    stunting_raw = pd.read_csv(stunting_file)
    stunting = stunting_raw.copy()
    stunting['provinsi'] = stunting['nama_provinsi'].str.upper().str.strip()
    stunting = stunting[stunting['provinsi'].isin(VALID_34)]
    stunting = stunting[['provinsi', 'prevalensi_balita_stunting', 'tahun']].copy()
    stunting.columns = ['provinsi', 'prevalensi_stunting', 'tahun']

    kemiskinan_filenames = {
        2021: "Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2021.csv",
        2022: "Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2022.csv",
        2023: "Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2023.csv",
        2024: "Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2024.csv",
    }
    kemiskinan_frames = []
    for year, fname in kemiskinan_filenames.items():
        path = resolve_path(fname, "dataset/raw")
        if not os.path.exists(path):
            st.error(f"❌ File tidak ditemukan: `{path}`")
            st.stop()
        kdf = pd.read_csv(path)
        kdf.columns = kdf.columns.str.strip()
        kdf['provinsi'] = kdf['Provinsi'].str.upper().str.strip()
        kdf = kdf[kdf['provinsi'].notna()]
        kdf = kdf[~kdf['provinsi'].isin(JUNK_ROWS)]
        kdf = kdf[kdf['provinsi'].isin(VALID_34)]
        kdf = kdf[['provinsi',
                    'Garis Kemiskinan - Maret (Rp)',
                    'Jumlah Penduduk Miskin - Maret (ribu) (Ribu)',
                    'Persentase Penduduk Miskin - Maret']].copy()
        kdf.columns = ['provinsi', 'garis_kemiskinan', 'jumlah_miskin_ribu', 'persen_miskin']
        kdf['tahun'] = year
        for col in ['garis_kemiskinan', 'jumlah_miskin_ribu', 'persen_miskin']:
            kdf[col] = pd.to_numeric(
                kdf[col].astype(str).str.replace(',', '').str.replace('...', ''),
                errors='coerce'
            )
        kemiskinan_frames.append(kdf)
    kemiskinan = pd.concat(kemiskinan_frames, ignore_index=True)

    sanitasi_filenames = {
        2021: "Persentase Rumah Tangga menurut Provinsi dan Memiliki Akses terhadap Sanitasi Layak, 2021.csv",
        2022: "Persentase Rumah Tangga menurut Provinsi dan Memiliki Akses terhadap Sanitasi Layak, 2022.csv",
        2023: "Persentase Rumah Tangga menurut Provinsi dan Memiliki Akses terhadap Sanitasi Layak, 2023.csv",
        2024: "Persentase Rumah Tangga menurut Provinsi dan Memiliki Akses terhadap Sanitasi Layak, 2024.csv",
    }
    sanitasi_frames = []
    for year, fname in sanitasi_filenames.items():
        path = resolve_path(fname, "dataset/raw")
        if not os.path.exists(path):
            st.error(f"❌ File tidak ditemukan: `{path}`")
            st.stop()
        sdf = pd.read_csv(path, header=None, skiprows=3)
        sdf.columns = ['provinsi', 'persen_sanitasi']
        sdf['provinsi'] = sdf['provinsi'].astype(str).str.upper().str.strip()
        sdf['provinsi'] = sdf['provinsi'].replace(SANITASI_NAME_MAP)
        sdf['persen_sanitasi'] = pd.to_numeric(
            sdf['persen_sanitasi'].astype(str).str.replace(',', '').str.replace('-', ''),
            errors='coerce'
        )
        sdf = sdf[sdf['provinsi'].isin(VALID_34)]
        sdf = sdf.dropna(subset=['persen_sanitasi'])
        sdf['tahun'] = year
        sanitasi_frames.append(sdf)
    sanitasi = pd.concat(sanitasi_frames, ignore_index=True)

    df = pd.merge(stunting, kemiskinan, on=['provinsi', 'tahun'], how='inner')
    df = pd.merge(df, sanitasi[['provinsi', 'tahun', 'persen_sanitasi']], on=['provinsi', 'tahun'], how='inner')
    df = df.sort_values(['tahun', 'provinsi']).reset_index(drop=True)

    return df, None

@st.cache_resource
def load_or_train_models(df):
    model_reg = None
    for reg_name in ["model_regresi_stunting.pkl", "model_regresi_multiple.pkl", "model_regresi_simple.pkl"]:
        reg_path = resolve_path(reg_name, "models")
        if os.path.exists(reg_path):
            try:
                if joblib is not None:
                    model_reg = joblib.load(reg_path)
                else:
                    with open(reg_path, "rb") as f:
                        model_reg = pickle.load(f)
                break
            except Exception as e:
                st.warning(f"Gagal memuat {reg_path}: {e}")

    if model_reg is None:
        model_reg = LinearRegression()
        X_cols = ['persen_miskin', 'persen_sanitasi'] if 'persen_sanitasi' in df.columns else ['persen_miskin']
        model_reg.fit(df[X_cols].values, df['prevalensi_stunting'].values)

    agg_dict = {
        'stunting_mean': ('prevalensi_stunting', 'mean'),
        'miskin_mean': ('persen_miskin', 'mean'),
    }
    if 'persen_sanitasi' in df.columns:
        agg_dict['sanitasi_mean'] = ('persen_sanitasi', 'mean')

    df_avg = df.groupby('provinsi').agg(**agg_dict).reset_index()

    clust_cols = ['miskin_mean', 'stunting_mean']
    if 'sanitasi_mean' in df_avg.columns:
        clust_cols.append('sanitasi_mean')

    X_clust = df_avg[clust_cols].values
    scaler_clust = StandardScaler()
    X_scaled = scaler_clust.fit_transform(X_clust)

    km_final = None
    model_loaded = False
    for km_name in ["model_kmeans_stunting.pkl", "model_kmeans.pkl"]:
        km_path = resolve_path(km_name, "models")
        if os.path.exists(km_path):
            try:
                if joblib is not None:
                    km_data = joblib.load(km_path)
                else:
                    with open(km_path, "rb") as f:
                        km_data = pickle.load(f)
                
                if isinstance(km_data, dict) and 'kmeans' in km_data and 'scaler' in km_data:
                    km_final = km_data['kmeans']
                    scaler_clust = km_data['scaler']
                else:
                    km_final = km_data
                X_scaled = scaler_clust.transform(X_clust)
                model_loaded = True
                break
            except Exception as e:
                st.warning(f"Gagal memuat {km_path}: {e}")

    if not model_loaded:
        silhouettes = []
        K_range = range(2, 9)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            silhouettes.append(silhouette_score(X_scaled, labels))
        best_k = list(K_range)[np.argmax(silhouettes)]
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        km_final.fit(X_scaled)

    df_avg['cluster'] = km_final.predict(X_scaled)
    best_k = km_final.n_clusters

    centroids = scaler_clust.inverse_transform(km_final.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=clust_cols)
    centroid_df['cluster'] = range(best_k)
    centroid_df = centroid_df.sort_values('stunting_mean', ascending=False)

    label_map = {}
    for i, (_, row) in enumerate(centroid_df.iterrows()):
        if i == 0:
            label_map[int(row['cluster'])] = 'Risiko Tinggi'
        elif i == best_k - 1:
            label_map[int(row['cluster'])] = 'Risiko Rendah'
        else:
            label_map[int(row['cluster'])] = 'Risiko Menengah'

    df_avg['label_cluster'] = df_avg['cluster'].map(label_map)
    sil_final = silhouette_score(X_scaled, df_avg['cluster'])

    return model_reg, km_final, scaler_clust, df_avg, sil_final, clust_cols

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    df, df_avg_loaded = load_data()
    model_reg, km_final, scaler_clust, df_avg, sil_final, clust_cols = load_or_train_models(df)
    
    if df_avg_loaded is not None:
        rename_map = {
            'prevalensi_stunting': 'stunting_mean',
            'persen_miskin': 'miskin_mean',
            'persen_sanitasi': 'sanitasi_mean',
            'cluster_label': 'label_cluster'
        }
        df_avg_loaded = df_avg_loaded.rename(columns=rename_map)
        df_avg = df_avg_loaded
            
    st.session_state["df"] = df
    st.session_state["model_reg"] = model_reg
    st.session_state["km_final"] = km_final
    st.session_state["scaler_clust"] = scaler_clust
    st.session_state["df_avg"] = df_avg
    st.session_state["sil_final"] = sil_final
    st.session_state["clust_cols"] = clust_cols
    st.session_state["has_sanitasi"] = 'persen_sanitasi' in df.columns

df = st.session_state["df"]
model_reg = st.session_state["model_reg"]
km_final = st.session_state["km_final"]
scaler_clust = st.session_state["scaler_clust"]
df_avg = st.session_state["df_avg"]
sil_final = st.session_state["sil_final"]
clust_cols = st.session_state["clust_cols"]
has_sanitasi = st.session_state["has_sanitasi"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("# Analisis Pengaruh Kemiskinan dan Sanitasi terhadap Prevalensi Stunting di Indonesia ")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Overview", "📊 Dashboard & Tren", "🔮 Prediksi Stunting", "🗺️ Pemetaan Zona Risiko"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Tentang Aplikasi")
st.sidebar.markdown("""
**CRISP-DM Fase 6: Deployment**

Analisis pengaruh **kemiskinan**
dan **sanitasi** terhadap
prevalensi stunting di 34
provinsi Indonesia (2021-2024).

- 📊 Dashboard Deskriptif
- 🔮 Prediksi (Regresi Multiple)
- 🗺️ Zona Risiko (K-Means 3D)
""")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align:center; opacity:0.5; font-size:0.8rem;'>"
    "Data Mining · Kelompok 5<br>Wahyu Jum’ah Maulidan <br>Moh. Harsa Ilham Akmaluddin<br>Muhammad Fairuz Zaki</p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN MENU DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
if menu == "🏠 Overview":
    
    # Page Config
    
    # Apply global styles
    
    # Warm-up Session State
    # Sidebar Info
    
    # Main Page Header
    st.markdown('<h1 class="main-header">📊 Analisis Kemiskinan & Sanitasi Terhadap Stunting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">CRISP-DM Fase 6: Deployment — Sistem Informasi & Pemodelan Prediktif Spasial</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Selamat datang di Aplikasi **Analisis Pengaruh Kemiskinan dan Sanitasi Terhadap Prevalensi Stunting di Indonesia**. 
    Aplikasi ini dikembangkan menggunakan metodologi standar industri **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) 
    untuk membantu pembuat kebijakan memahami faktor sosio-ekonomi dan lingkungan yang mempengaruhi stunting serta menentukan prioritas wilayah intervensi.
    """)
    
    # Visual Grid of Variables
    st.markdown("### 🔬 Variabel Penelitian")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color:#e5393515; padding:20px; border-radius:10px; border-left: 5px solid #e53935; height:150px;">
            <h4 style="margin-top:0; color:#e53935; margin-bottom:8px;">Prevalensi Balita Stunting (Y)</h4>
            <p style="font-size:0.85rem; margin-bottom:0; color:#ddd; line-height:1.4;">
                Variabel terikat yang mengukur persentase anak balita dengan tinggi badan menurut umur di bawah standar (sumber: Kemenkes RI).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:#1565c015; padding:20px; border-radius:10px; border-left: 5px solid #1565c0; height:150px;">
            <h4 style="margin-top:0; color:#1565c0; margin-bottom:8px;">Persentase Penduduk Miskin (X1)</h4>
            <p style="font-size:0.85rem; margin-bottom:0; color:#ddd; line-height:1.4;">
                Variabel bebas pertama yang merepresentasikan persentase penduduk di bawah garis kemiskinan BPS per Maret (sumber: BPS RI).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color:#2e7d3215; padding:20px; border-radius:10px; border-left: 5px solid #2e7d32; height:150px;">
            <h4 style="margin-top:0; color:#2e7d32; margin-bottom:8px;">Akses Sanitasi Layak (X2)</h4>
            <p style="font-size:0.85rem; margin-bottom:0; color:#ddd; line-height:1.4;">
                Variabel bebas kedua yang mengukur persentase rumah tangga yang memiliki akses sanitasi layak secara berkelanjutan (sumber: BPS RI).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Methodology Accordion
    st.markdown("---")
    st.markdown("### 🔄 Metodologi CRISP-DM")
    
    with st.expander("1. Business Understanding (Pemahaman Bisnis)"):
        st.markdown("""
        Menganalisis kebutuhan penanganan stunting di Indonesia dan merumuskan hipotesis hubungan kemiskinan dan sanitasi terhadap stunting.
        """)
    
    with st.expander("2. Data Understanding (Pemahaman Data)"):
        st.markdown("""
        Mengumpulkan dan mengeksplorasi data prevalensi stunting, data kemiskinan, dan data sanitasi per provinsi selama periode 2021-2024.
        """)
    
    with st.expander("3. Data Preparation (Persiapan Data)"):
        st.markdown("""
        Membersihkan data mentah dari inkonsistensi nama provinsi, mengonversi nilai ilegal, menyaring 34 provinsi utama, dan menggabungkan data (*merging*).
        """)
    
    with st.expander("4. Modeling (Pemodelan)"):
        st.markdown("""
        Menerapkan metode **Multiple Linear Regression** untuk pemodelan prediktif stunting dan **K-Means Clustering 3D** untuk pemetaan zona risiko daerah.
        """)
    
    with st.expander("5. Evaluation (Evaluasi)"):
        st.markdown("""
        Menguji stabilitas model regresi menggunakan metode *Temporal Validation* (uji coba data latih vs data uji) dan mengukur efisiensi klasterisasi menggunakan *Silhouette Score*.
        """)
    
    with st.expander("6. Deployment (Penyebaran)"):
        st.markdown("""
        Mengimplementasikan model ke dalam aplikasi dashboard Streamlit interaktif yang dapat diakses oleh instansi pemerintah dan akademisi (halaman ini).
        """)
    
    st.info("💡 **Tips:** Gunakan panel navigasi di sebelah kiri untuk berpindah halaman ke **Dashboard & Tren**, **Prediksi Stunting**, atau **Pemetaan Zona Risiko**.")

elif menu == "📊 Dashboard & Tren":
    
    
    # Apply global styles
    
    # Session State Check
    
    # Sidebar Info
    
    # Page Header
    st.markdown('<h1 class="main-header">📊 Dashboard & Tren Perkembangan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Eksplorasi data kemiskinan, sanitasi, dan stunting 34 provinsi di Indonesia (2021–2024)</p>', unsafe_allow_html=True)
    
    # ── Filter ────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        tahun_filter = st.multiselect(
            "📅 Pilih Tahun", options=sorted(df['tahun'].unique()),
            default=sorted(df['tahun'].unique()),
        )
    with col_f2:
        prov_filter = st.multiselect(
            "🏛️ Pilih Provinsi (kosongkan = semua)",
            options=sorted(df['provinsi'].unique()),
        )
    
    df_filtered = df[df['tahun'].isin(tahun_filter)]
    if prov_filter:
        df_filtered = df_filtered[df_filtered['provinsi'].isin(prov_filter)]
    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")
        st.stop()
    
    # ── Metric Cards ──────────────────────────────────────────────────────
    st.markdown("### 🔢 Ringkasan Statistik")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📍 Jumlah Provinsi", f"{df_filtered['provinsi'].nunique()}")
    c2.metric("📈 Rata-rata Stunting", f"{df_filtered['prevalensi_stunting'].mean():.1f}%")
    c3.metric("💰 Rata-rata Kemiskinan", f"{df_filtered['persen_miskin'].mean():.1f}%")
    
    if has_sanitasi:
        c4.metric("🚿 Rata-rata Sanitasi", f"{df_filtered['persen_sanitasi'].mean():.1f}%")
        
        # Calculate target achievement
        rt_lulus = df_filtered[df_filtered['prevalensi_stunting'] < 14.0]['provinsi'].nunique()
        total_prov_filtered = df_filtered['provinsi'].nunique()
        c5.metric("🎯 Provinsi Target (<14%)", f"{rt_lulus} / {total_prov_filtered}")
    else:
        c4.metric("🎯 Provinsi Target (<14%)", f"{df_filtered[df_filtered['prevalensi_stunting'] < 14.0]['provinsi'].nunique()}")
    
    st.markdown("---")
    
    # ── Tabs Visualisasi ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tren Nasional", "🔵 Scatter Plot", "🏛️ Perbandingan Provinsi", "🌡️ Heatmap Korelasi"])
    
    with tab1:
        st.markdown("#### Tren Rata-rata Nasional (2021-2024)")
        
        tren_agg = {
            'stunting_mean': ('prevalensi_stunting', 'mean'),
            'stunting_std': ('prevalensi_stunting', 'std'),
            'miskin_mean': ('persen_miskin', 'mean'),
        }
        if has_sanitasi:
            tren_agg['sanitasi_mean'] = ('persen_sanitasi', 'mean')
        
        # Group by year for national averages
        tren_df = df[df['tahun'].isin(tahun_filter)]
        if prov_filter:
            tren_df = tren_df[tren_df['provinsi'].isin(prov_filter)]
            
        tren = tren_df.groupby('tahun').agg(**tren_agg).reset_index()
    
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
    
        ax1.plot(tren['tahun'], tren['stunting_mean'], 'o-',
                 color='#e53935', linewidth=3, markersize=10, label='Stunting (%)', zorder=5)
                 
        # Add error band if multiple years
        if len(tren) > 1 and not tren['stunting_std'].isna().all():
            ax1.fill_between(tren['tahun'],
                             tren['stunting_mean'] - tren['stunting_std'].fillna(0),
                             tren['stunting_mean'] + tren['stunting_std'].fillna(0),
                             alpha=0.12, color='#e53935')
                             
        ax2.plot(tren['tahun'], tren['miskin_mean'], 's--',
                 color='#1565c0', linewidth=3, markersize=10, label='Kemiskinan (%)', zorder=5)
    
        if has_sanitasi and 'sanitasi_mean' in tren.columns:
            ax2.plot(tren['tahun'], tren['sanitasi_mean'], 'D-.',
                     color='#2e7d32', linewidth=3, markersize=10, label='Sanitasi (%)', zorder=5)
    
        ax1.set_xlabel('Tahun', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Prevalensi Stunting (%)', color='#e53935', fontsize=11)
        ax2.set_ylabel('Kemiskinan / Sanitasi (%)', fontsize=11)
        ax1.set_xticks(tren['tahun'].tolist())
        ax1.set_title('Tren Perkembangan Indikator Nasional', fontsize=14, fontweight='bold', pad=15)
    
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#ccc')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
        if len(tren) >= 2:
            delta_s = tren['stunting_mean'].iloc[-1] - tren['stunting_mean'].iloc[0]
            delta_m = tren['miskin_mean'].iloc[-1] - tren['miskin_mean'].iloc[0]
            arah_s = "turun" if delta_s < 0 else "naik"
            arah_m = "turun" if delta_m < 0 else "naik"
            insight = (f"📌 **Insight:** Prevalensi stunting rata-rata {arah_s} sebesar **{abs(delta_s):.1f}pp** "
                       f"dan kemiskinan {arah_m} sebesar **{abs(delta_m):.1f}pp**.")
            if has_sanitasi and 'sanitasi_mean' in tren.columns:
                delta_san = tren['sanitasi_mean'].iloc[-1] - tren['sanitasi_mean'].iloc[0]
                arah_san = "naik" if delta_san > 0 else "turun"
                insight += f" Akses sanitasi layak {arah_san} sebesar **{abs(delta_san):.1f}pp**."
            st.info(insight)
    
    with tab2:
        st.markdown("#### Scatter Plot Korelasi")
        scatter_var = st.selectbox("Pilih variabel X:", ["persen_miskin", "persen_sanitasi"] if has_sanitasi else ["persen_miskin"])
        x_label = "% Penduduk Miskin" if scatter_var == "persen_miskin" else "Akses Sanitasi Layak (%)"
    
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_year = {2021: '#1565c0', 2022: '#ef6c00', 2023: '#2e7d32', 2024: '#6a1b9a'}
        for year in sorted(df_filtered['tahun'].unique()):
            sub = df_filtered[df_filtered['tahun'] == year]
            ax.scatter(sub[scatter_var], sub['prevalensi_stunting'],
                       color=colors_year.get(year, '#333'), label=str(year),
                       alpha=0.75, s=65, edgecolors='white', linewidth=0.6, zorder=3)
    
        r_sc, p_sc = pearsonr(df_filtered[scatter_var], df_filtered['prevalensi_stunting'])
        m, b = np.polyfit(df_filtered[scatter_var], df_filtered['prevalensi_stunting'], 1)
        x_line = np.linspace(df_filtered[scatter_var].min(), df_filtered[scatter_var].max(), 100)
        ax.plot(x_line, m * x_line + b, 'k--', linewidth=2.5, alpha=0.7, label=f'Regresi (r={r_sc:.3f})')
    
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Prevalensi Stunting (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Korelasi {x_label} vs Stunting', fontsize=14, fontweight='bold')
        ax.legend(title='Tahun', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.03, 0.95, f'r = {r_sc:.3f}\np = {p_sc:.2e}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85, edgecolor='#ccc'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
        kekuatan = "**sangat kuat**" if abs(r_sc) >= 0.7 else "**kuat**" if abs(r_sc) >= 0.5 else "**sedang**" if abs(r_sc) >= 0.3 else "**lemah**"
        arah = "positif" if r_sc > 0 else "negatif"
        st.success(f"🔗 Korelasi Pearson: **r = {r_sc:.3f}** — hubungan {kekuatan} dan {arah}. "
                   f"{'Signifikan (p < 0.05)' if p_sc < 0.05 else 'Tidak signifikan'}.")
    
    with tab3:
        st.markdown("#### Perbandingan Stunting, Kemiskinan & Sanitasi per Provinsi")
        metric_choice = st.selectbox("Urutkan berdasarkan:", ["prevalensi_stunting", "persen_miskin"] + (["persen_sanitasi"] if has_sanitasi else []))
    
        prov_agg = {'stunting': ('prevalensi_stunting', 'mean'), 'miskin': ('persen_miskin', 'mean')}
        if has_sanitasi:
            prov_agg['sanitasi'] = ('persen_sanitasi', 'mean')
        prov_avg = df_filtered.groupby('provinsi').agg(**prov_agg).reset_index()
    
        sort_col = 'stunting' if metric_choice == 'prevalensi_stunting' else ('miskin' if metric_choice == 'persen_miskin' else 'sanitasi')
        prov_avg = prov_avg.sort_values(sort_col, ascending=True)
    
        fig, ax = plt.subplots(figsize=(12, max(8, len(prov_avg) * 0.38)))
        y_pos = np.arange(len(prov_avg))
        n_bars = 3 if has_sanitasi else 2
        bar_width = 0.8 / n_bars
    
        ax.barh(y_pos - bar_width, prov_avg['stunting'], bar_width,
                color='#e53935', alpha=0.85, label='Stunting (%)', edgecolor='white')
        ax.barh(y_pos, prov_avg['miskin'], bar_width,
                color='#1565c0', alpha=0.85, label='Kemiskinan (%)', edgecolor='white')
        if has_sanitasi:
            ax.barh(y_pos + bar_width, prov_avg['sanitasi'], bar_width,
                    color='#2e7d32', alpha=0.85, label='Sanitasi (%)', edgecolor='white')
    
        ax.set_yticks(y_pos)
        ax.set_yticklabels([p.title() for p in prov_avg['provinsi']], fontsize=9)
        ax.set_xlabel('Persentase (%)', fontsize=11, fontweight='bold')
        ax.set_title('Rata-rata Kinerja per Provinsi', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab4:
        st.markdown("#### Heatmap Korelasi Antar Variabel")
        num_cols = ['prevalensi_stunting', 'persen_miskin']
        if has_sanitasi:
            num_cols.append('persen_sanitasi')
        num_cols.extend([c for c in ['garis_kemiskinan', 'jumlah_miskin_ribu'] if c in df_filtered.columns])
        corr_matrix = df_filtered[num_cols].corr()
    
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        labels_map = {
            'prevalensi_stunting': 'Stunting', 'persen_miskin': '% Miskin',
            'persen_sanitasi': 'Sanitasi', 'garis_kemiskinan': 'Garis Kemiskinan',
            'jumlah_miskin_ribu': 'Jumlah Miskin'
        }
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    mask=mask, ax=ax, vmin=-1, vmax=1, square=True, linewidths=0.8,
                    annot_kws={'size': 12, 'weight': 'bold'}, cbar_kws={'shrink': 0.8})
        ax.set_xticklabels([labels_map.get(c, c) for c in num_cols], rotation=25, ha='right')
        ax.set_yticklabels([labels_map.get(c, c) for c in num_cols], rotation=0)
        ax.set_title('Heatmap Korelasi Pearson', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    with st.expander("📋 Lihat Tabel Data Lengkap"):
        fmt = {'prevalensi_stunting': '{:.2f}', 'persen_miskin': '{:.2f}',
               'garis_kemiskinan': '{:,.0f}', 'jumlah_miskin_ribu': '{:.2f}'}
        if has_sanitasi:
            fmt['persen_sanitasi'] = '{:.2f}'
        st.dataframe(df_filtered.style.format(fmt), use_container_width=True, height=400)

elif menu == "🔮 Prediksi Stunting":
    
    
    # Apply global styles
    
    # Session State Check
    
    # Sidebar Info
    
    # Page Header
    st.markdown('<h1 class="main-header">🔮 Sistem Prediksi Prevalensi Stunting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prediksi stunting menggunakan model Regresi Linear Berganda (Kemiskinan + Sanitasi)</p>', unsafe_allow_html=True)
    
    # Determine model features
    if has_sanitasi and model_reg.coef_.shape[0] >= 2:
        X_features = df[['persen_miskin', 'persen_sanitasi']].values
        is_multiple = True
    else:
        X_features = df[['persen_miskin']].values
        is_multiple = False
    
    y_all = df['prevalensi_stunting'].values
    y_pred_all = model_reg.predict(X_features)
    r2 = r2_score(y_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_all, y_pred_all))
    mae = mean_absolute_error(y_all, y_pred_all)
    
    # ── Info Model ────────────────────────────────────────────────────────
    st.markdown("### 📐 Informasi Model Regresi Linear " + ("Berganda" if is_multiple else "Sederhana"))
    
    if is_multiple:
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("R² Score", f"{r2:.4f}")
        col_m2.metric("RMSE", f"{rmse:.2f}%")
        col_m3.metric("MAE", f"{mae:.2f}%")
        col_m4.metric("β₁ (Kemiskinan)", f"{model_reg.coef_[0]:.4f}")
        col_m5.metric("β₂ (Sanitasi)", f"{model_reg.coef_[1]:.4f}")
        st.code(f"Persamaan:  Stunting = {model_reg.intercept_:.2f} + {model_reg.coef_[0]:.2f} × Kemiskinan + ({model_reg.coef_[1]:.2f}) × Sanitasi", language="text")
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("R² Score", f"{r2:.4f}")
        col_m2.metric("RMSE", f"{rmse:.2f}%")
        col_m3.metric("MAE", f"{mae:.2f}%")
        col_m4.metric("β₁ (Kemiskinan)", f"{model_reg.coef_[0]:.4f}")
        st.code(f"Persamaan:  Stunting = {model_reg.intercept_:.2f} + {model_reg.coef_[0]:.2f} × Kemiskinan", language="text")
    
    st.markdown("---")
    
    # ── Input User ────────────────────────────────────────────────────────
    st.markdown("### 🎛️ Masukkan Data")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        input_method = st.radio("Metode Input:", ["🔘 Slider", "⌨️ Ketik Manual"], horizontal=True)
    
        if input_method == "🔘 Slider":
            input_miskin = st.slider("Persentase Penduduk Miskin (%)",
                                     min_value=0.0, max_value=35.0, value=10.0, step=0.1)
            if is_multiple:
                input_sanitasi = st.slider("Akses Sanitasi Layak (%)",
                                           min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        else:
            input_miskin = st.number_input("Persentase Penduduk Miskin (%)",
                                           min_value=0.0, max_value=50.0, value=10.0, step=0.1)
            if is_multiple:
                input_sanitasi = st.number_input("Akses Sanitasi Layak (%)",
                                                  min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    
        if is_multiple:
            pred_stunting = model_reg.predict([[input_miskin, input_sanitasi]])[0]
        else:
            pred_stunting = model_reg.predict([[input_miskin]])[0]
        pred_stunting = max(0, pred_stunting)
    
    with col_result:
        if pred_stunting >= 30:
            kat, kat_desc, color_class = "🔴 SANGAT TINGGI", "Memerlukan intervensi gizi darurat", "#e53935"
        elif pred_stunting >= 20:
            kat, kat_desc, color_class = "🟠 TINGGI", "Masih di atas target nasional 14%", "#ef6c00"
        elif pred_stunting >= 14:
            kat, kat_desc, color_class = "🟡 SEDANG", "Mendekati target nasional 14%", "#f9a825"
        else:
            kat, kat_desc, color_class = "🟢 RENDAH", "Sudah memenuhi target nasional", "#2e7d32"
    
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, {color_class}dd, {color_class}88);">
            <p style="font-size: 0.9rem; margin-bottom: 12px;">Estimasi Prevalensi Stunting</p>
            <h2>{pred_stunting:.1f}%</h2>
            <p style="font-size: 1.1rem; font-weight: 700; margin-top: 12px;">{kat}</p>
            <p style="font-size: 0.85rem;">{kat_desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Visualisasi Prediksi ──────────────────────────────────────────────
    st.markdown("### 📊 Visualisasi Posisi Prediksi")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Scatter
    colors_year = {2021: '#1565c0', 2022: '#ef6c00', 2023: '#2e7d32', 2024: '#6a1b9a'}
    for year in sorted(df['tahun'].unique()):
        sub = df[df['tahun'] == year]
        axes[0].scatter(sub['persen_miskin'], sub['prevalensi_stunting'],
                        color=colors_year.get(year, '#333'), label=str(year),
                        alpha=0.55, s=45, edgecolors='white', linewidth=0.5)
    
    x_line = np.linspace(0, max(35, input_miskin + 5), 100)
    if is_multiple:
        y_line = model_reg.predict(np.column_stack([x_line, np.full_like(x_line, input_sanitasi if is_multiple else 80)]))
    else:
        y_line = model_reg.predict(x_line.reshape(-1, 1))
    axes[0].plot(x_line, y_line, 'k--', linewidth=2, alpha=0.6)
    axes[0].scatter([input_miskin], [pred_stunting], color='#e53935', s=250, zorder=10,
                    edgecolors='white', linewidth=3, marker='*', label='Prediksi Anda')
    axes[0].annotate(f'({input_miskin:.1f}%, {pred_stunting:.1f}%)',
                     (input_miskin, pred_stunting), xytext=(15, 15),
                     textcoords='offset points', fontsize=11, fontweight='bold', color='#e53935',
                     arrowprops=dict(arrowstyle='->', color='#e53935', lw=2))
    axes[0].set_xlabel('% Penduduk Miskin', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Prevalensi Stunting (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Posisi Prediksi pada Scatter Plot', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Right: Gauge
    categories = ['Rendah\n(< 14%)', 'Sedang\n(14-20%)', 'Tinggi\n(20-30%)', 'Sangat Tinggi\n(> 30%)']
    cat_colors = ['#2e7d32', '#f9a825', '#ef6c00', '#e53935']
    cat_ranges = [14, 6, 10, 10]
    bottom = 0
    for cat, col, rng in zip(categories, cat_colors, cat_ranges):
        axes[1].barh(0, rng, left=bottom, height=0.5, color=col, alpha=0.4, edgecolor='white')
        axes[1].text(bottom + rng / 2, 0, cat, ha='center', va='center', fontsize=8, fontweight='bold')
        bottom += rng
    axes[1].axvline(pred_stunting, color='#e53935', linewidth=3, zorder=5)
    axes[1].scatter([pred_stunting], [0], color='#e53935', s=200, zorder=10, edgecolors='white', linewidth=2, marker='v')
    axes[1].text(pred_stunting, 0.35, f'{pred_stunting:.1f}%', ha='center', fontsize=14, fontweight='bold', color='#e53935')
    axes[1].set_xlim(0, 40)
    axes[1].set_ylim(-0.5, 0.6)
    axes[1].set_xlabel('Prevalensi Stunting (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Posisi pada Skala Kategori', fontsize=12, fontweight='bold')
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ── Interpretasi ──────────────────────────────────────────────────────
    st.markdown("### 💡 Interpretasi")
    if is_multiple:
        st.info(f"""
        **Dengan kemiskinan {input_miskin:.1f}% dan sanitasi {input_sanitasi:.1f}%**, model regresi berganda
        memprediksi prevalensi stunting sebesar **{pred_stunting:.1f}%**.
    
        - Persamaan: `stunting = {model_reg.intercept_:.2f} + {model_reg.coef_[0]:.2f} × kemiskinan + ({model_reg.coef_[1]:.2f}) × sanitasi`
        - Setiap kenaikan **1pp kemiskinan** → stunting naik **~{model_reg.coef_[0]:.2f}pp**
        - Setiap kenaikan **1pp sanitasi** → stunting {'turun' if model_reg.coef_[1] < 0 else 'naik'} **~{abs(model_reg.coef_[1]):.2f}pp**
        - Target nasional RPJMN 2024: **14%**. Prediksi ini {'**belum memenuhi**' if pred_stunting >= 14 else '**sudah memenuhi**'} target.
        """)
    else:
        st.info(f"""
        **Dengan kemiskinan {input_miskin:.1f}%**, model regresi
        memprediksi prevalensi stunting sebesar **{pred_stunting:.1f}%**.
    
        - Persamaan: `stunting = {model_reg.intercept_:.2f} + {model_reg.coef_[0]:.2f} × kemiskinan`
        - Setiap kenaikan **1pp kemiskinan** → stunting naik **~{model_reg.coef_[0]:.2f}pp**
        - Target nasional RPJMN 2024: **14%**. Prediksi ini {'**belum memenuhi**' if pred_stunting >= 14 else '**sudah memenuhi**'} target.
        """)

elif menu == "🗺️ Pemetaan Zona Risiko":
    
    
    # Apply global styles
    
    # Session State Check
    
    # Sidebar Info
    
    # Page Header
    st.markdown('<h1 class="main-header">🗺️ Pemetaan Zona Risiko Provinsi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Klasifikasi provinsi menggunakan K-Means Clustering 3 dimensi (Kemiskinan, Sanitasi, Stunting)</p>', unsafe_allow_html=True)
    
    best_k = km_final.n_clusters
    
    st.markdown("### 🔬 Informasi Model K-Means")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Cluster (k)", f"{best_k}")
    c2.metric("Silhouette Score", f"{sil_final:.4f}")
    c3.metric("Jumlah Provinsi", f"{len(df_avg)}")
    c4.metric("Dimensi Fitur", f"{len(clust_cols)}")
    
    st.markdown("---")
    
    # ── Cluster Cards ─────────────────────────────────────────────────────
    st.markdown("### 🏷️ Profil Setiap Cluster")
    
    cluster_labels_sorted = ['Risiko Tinggi', 'Risiko Menengah', 'Risiko Rendah']
    cluster_labels_available = [l for l in cluster_labels_sorted if l in df_avg['label_cluster'].values]
    for l in sorted(df_avg['label_cluster'].unique()):
        if l not in cluster_labels_available:
            cluster_labels_available.append(l)
    
    cols_cluster = st.columns(len(cluster_labels_available))
    for col, label in zip(cols_cluster, cluster_labels_available):
        sub = df_avg[df_avg['label_cluster'] == label]
        count = len(sub)
        st_mean = sub['stunting_mean'].mean()
        m_mean = sub['miskin_mean'].mean()
    
        if 'Tinggi' in label and 'Menengah' not in label:
            css_class, emoji = "risk-high", "🔴"
        elif 'Menengah' in label:
            css_class, emoji = "risk-medium", "🟡"
        else:
            css_class, emoji = "risk-low", "🟢"
    
        san_html = ""
        if 'sanitasi_mean' in sub.columns:
            san_mean = sub['sanitasi_mean'].mean()
            san_html = f"<p>Sanitasi: <strong>{san_mean:.1f}%</strong></p>"
    
        with col:
            st.markdown(f"""
            <div class="risk-card {css_class}">
                <h3>{emoji} {label}</h3>
                <p><strong>{count}</strong> Provinsi</p>
                <p>Stunting: <strong>{st_mean:.1f}%</strong></p>
                <p>Kemiskinan: <strong>{m_mean:.1f}%</strong></p>
                {san_html}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Visualisasi ───────────────────────────────────────────────────────
    st.markdown("### 📊 Visualisasi Clustering")
    
    tab_c0, tab_c1, tab_c2, tab_c3 = st.tabs(["🗺️ Peta Geografis", "🔵 Scatter Plot", "📋 Daftar Provinsi", "📊 Profil Cluster"])
    
    cluster_colors_map = {
        'Risiko Tinggi': '#e53935',
        'Risiko Menengah': '#ef6c00',
        'Risiko Rendah': '#2e7d32',
    }
    
    with tab_c0:
        st.markdown("#### Peta Persebaran Zona Risiko")
        st.info("💡 **Tips Interaktif:** Arahkan kursor (hover) ke titik pada peta untuk melihat detail provinsi.")
    
        prov_coords = {
            'ACEH': [4.6951, 96.7494], 'SUMATERA UTARA': [2.1154, 99.5451],
            'SUMATERA BARAT': [-0.7399, 100.8000], 'RIAU': [0.2933, 101.7068],
            'JAMBI': [-1.6101, 103.6131], 'SUMATERA SELATAN': [-3.3194, 104.9147],
            'BENGKULU': [-3.7928, 102.2608], 'LAMPUNG': [-4.5586, 105.1704],
            'KEPULAUAN BANGKA BELITUNG': [-2.7411, 106.4406],
            'KEPULAUAN RIAU': [3.9456, 108.1429],
            'DKI JAKARTA': [-6.2088, 106.8456], 'JAWA BARAT': [-6.9204, 107.6046],
            'JAWA TENGAH': [-7.1509, 110.1403], 'DI YOGYAKARTA': [-7.7956, 110.3695],
            'JAWA TIMUR': [-7.5360, 112.2384], 'BANTEN': [-6.4058, 106.0640],
            'BALI': [-8.4095, 115.1889], 'NUSA TENGGARA BARAT': [-8.6529, 117.3616],
            'NUSA TENGGARA TIMUR': [-8.6574, 121.0794],
            'KALIMANTAN BARAT': [-0.2788, 111.4753],
            'KALIMANTAN TENGAH': [-1.6815, 113.3824],
            'KALIMANTAN SELATAN': [-3.0926, 115.2838],
            'KALIMANTAN TIMUR': [1.6406, 116.6415],
            'KALIMANTAN UTARA': [3.0731, 116.0414],
            'SULAWESI UTARA': [0.9997, 124.0902],
            'SULAWESI TENGAH': [-1.4300, 121.4456],
            'SULAWESI SELATAN': [-3.6688, 119.9740],
            'SULAWESI TENGGARA': [-4.1449, 122.1746],
            'GORONTALO': [0.6999, 122.4467],
            'SULAWESI BARAT': [-2.8441, 119.2321],
            'MALUKU': [-3.2385, 130.1453], 'MALUKU UTARA': [1.5709, 127.8088],
            'PAPUA BARAT': [-1.3361, 133.1747], 'PAPUA': [-4.2699, 138.0804]
        }
    
        df_map = df_avg.copy()
        df_map['lat'] = df_map['provinsi'].map(lambda x: prov_coords.get(x, [0,0])[0])
        df_map['lon'] = df_map['provinsi'].map(lambda x: prov_coords.get(x, [0,0])[1])
        df_map = df_map[df_map['lat'] != 0]
        df_map['Provinsi'] = df_map['provinsi'].str.title()
    
        hover_data_map = {
            "stunting_mean": ':.1f', "miskin_mean": ':.1f',
            "lat": False, "lon": False, "label_cluster": False
        }
        if 'sanitasi_mean' in df_map.columns:
            hover_data_map["sanitasi_mean"] = ':.1f'
    
        labels_map = {
            "label_cluster": "Zona Risiko", "stunting_mean": "Stunting (%)",
            "miskin_mean": "Kemiskinan (%)", "sanitasi_mean": "Sanitasi (%)"
        }
    
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", color="label_cluster",
            hover_name="Provinsi", hover_data=hover_data_map,
            color_discrete_map=cluster_colors_map,
            zoom=3.8, center={"lat": -2.0, "lon": 118.0},
            mapbox_style="carto-positron", labels=labels_map
        )
        fig_map.update_traces(marker=dict(size=14, opacity=0.85))
        fig_map.update_layout(
            margin={"r":0,"t":10,"l":0,"b":0},
            legend=dict(title=dict(text='Zona Risiko'), yanchor="top", y=0.95,
                        xanchor="left", x=0.05, bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#ddd", borderwidth=1)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    with tab_c1:
        st.markdown("#### Scatter Plot Clustering")
    
        if has_sanitasi and 'sanitasi_mean' in df_avg.columns:
            scatter_x = st.selectbox("Sumbu X:", ["miskin_mean", "sanitasi_mean"], format_func=lambda x: "Kemiskinan" if x == "miskin_mean" else "Sanitasi")
        else:
            scatter_x = "miskin_mean"
    
        fig, ax = plt.subplots(figsize=(12, 7))
        for label in cluster_labels_available:
            sub = df_avg[df_avg['label_cluster'] == label]
            color = cluster_colors_map.get(label, '#666')
            ax.scatter(sub[scatter_x], sub['stunting_mean'],
                       color=color, label=label, s=120, alpha=0.85,
                       edgecolors='white', linewidth=1, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(row['provinsi'].title(), (row[scatter_x], row['stunting_mean']),
                            fontsize=7, alpha=0.8, xytext=(4, 4), textcoords='offset points')
    
        x_label_sc = "Rata-rata % Kemiskinan" if scatter_x == "miskin_mean" else "Rata-rata Sanitasi (%)"
        ax.set_xlabel(x_label_sc, fontsize=12, fontweight='bold')
        ax.set_ylabel('Rata-rata Prevalensi Stunting (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering (k={best_k}, Silhouette={sil_final:.3f})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab_c2:
        for label in cluster_labels_available:
            sub = df_avg[df_avg['label_cluster'] == label].sort_values('stunting_mean', ascending=False)
            if 'Tinggi' in label and 'Menengah' not in label:
                emoji = "🔴"
            elif 'Menengah' in label:
                emoji = "🟡"
            else:
                emoji = "🟢"
    
            st.markdown(f"#### {emoji} {label} ({len(sub)} provinsi)")
            disp_cols = ['provinsi', 'stunting_mean', 'miskin_mean']
            disp_names = ['Provinsi', 'Stunting (%)', 'Kemiskinan (%)']
            if 'sanitasi_mean' in sub.columns:
                disp_cols.append('sanitasi_mean')
                disp_names.append('Sanitasi (%)')
    
            display_df = sub[disp_cols].copy()
            display_df.columns = disp_names
            display_df['Provinsi'] = display_df['Provinsi'].str.title()
            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
    
            fmt_disp = {'Stunting (%)': '{:.1f}', 'Kemiskinan (%)': '{:.1f}'}
            if 'Sanitasi (%)' in display_df.columns:
                fmt_disp['Sanitasi (%)'] = '{:.1f}'
            st.dataframe(display_df.style.format(fmt_disp), use_container_width=True)
    
    with tab_c3:
        st.markdown("#### Perbandingan Profil Antar Cluster")
        profile_agg = {
            'jumlah_provinsi': ('provinsi', 'count'),
            'stunting_mean': ('stunting_mean', 'mean'),
            'stunting_min': ('stunting_mean', 'min'),
            'stunting_max': ('stunting_mean', 'max'),
            'miskin_mean': ('miskin_mean', 'mean'),
        }
        if 'sanitasi_mean' in df_avg.columns:
            profile_agg['sanitasi_mean_avg'] = ('sanitasi_mean', 'mean')
        profile = df_avg.groupby('label_cluster').agg(**profile_agg).round(2)
        st.dataframe(profile, use_container_width=True)
    
        cluster_order = [l for l in cluster_labels_available if l in profile.index]
        n_bars_prof = 3 if 'sanitasi_mean' in df_avg.columns else 2
        fig, axes = plt.subplots(1, n_bars_prof, figsize=(5 * n_bars_prof, 4.5))
        if n_bars_prof == 2:
            axes = list(axes)
    
        bars1 = axes[0].bar(cluster_order, [profile.loc[l, 'stunting_mean'] for l in cluster_order],
                            color=[cluster_colors_map.get(l, '#666') for l in cluster_order],
                            alpha=0.85, edgecolor='white', width=0.5)
        axes[0].bar_label(bars1, fmt='%.1f%%', fontweight='bold', padding=3)
        axes[0].set_ylabel('Stunting (%)')
        axes[0].set_title('Rata-rata Stunting', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
    
        bars2 = axes[1].bar(cluster_order, [profile.loc[l, 'miskin_mean'] for l in cluster_order],
                            color=[cluster_colors_map.get(l, '#666') for l in cluster_order],
                            alpha=0.85, edgecolor='white', width=0.5)
        axes[1].bar_label(bars2, fmt='%.1f%%', fontweight='bold', padding=3)
        axes[1].set_ylabel('Kemiskinan (%)')
        axes[1].set_title('Rata-rata Kemiskinan', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
    
        if n_bars_prof == 3 and 'sanitasi_mean_avg' in profile.columns:
            bars3 = axes[2].bar(cluster_order, [profile.loc[l, 'sanitasi_mean_avg'] for l in cluster_order],
                                color=[cluster_colors_map.get(l, '#666') for l in cluster_order],
                                alpha=0.85, edgecolor='white', width=0.5)
            axes[2].bar_label(bars3, fmt='%.1f%%', fontweight='bold', padding=3)
            axes[2].set_ylabel('Sanitasi (%)')
            axes[2].set_title('Rata-rata Sanitasi', fontweight='bold')
            axes[2].grid(True, alpha=0.3, axis='y')
    
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # ── Rekomendasi Kebijakan ─────────────────────────────────────────────
    st.markdown("### 📌 Rekomendasi Kebijakan")
    
    risiko_tinggi = df_avg[df_avg['label_cluster'] == 'Risiko Tinggi']['provinsi'].tolist()
    if risiko_tinggi:
        prov_list = ", ".join([p.title() for p in risiko_tinggi])
        st.error(f"""
        **🔴 Provinsi Risiko Tinggi** memerlukan perhatian prioritas:
        {prov_list}
    
        **Rekomendasi:**
        - Alokasi anggaran darurat program penanganan stunting
        - Sinergi program pengentasan kemiskinan + perbaikan gizi + peningkatan sanitasi
        - Monitoring ketat melalui Posyandu dan PKH
        - Peningkatan akses air bersih, sanitasi layak, dan pangan bergizi
        """)
    
    risiko_menengah = df_avg[df_avg['label_cluster'].str.contains('Menengah', na=False)]['provinsi'].tolist()
    if risiko_menengah:
        prov_list_m = ", ".join([p.title() for p in risiko_menengah])
        st.warning(f"""
        **🟡 Provinsi Risiko Menengah:**
        {prov_list_m}
    
        **Rekomendasi:**
        - Penguatan program pencegahan stunting di tingkat desa
        - Perbaikan infrastruktur sanitasi dan air bersih
        - Monitoring berkala dan evaluasi program
        """)
    
    risiko_rendah = df_avg[df_avg['label_cluster'] == 'Risiko Rendah']['provinsi'].tolist()
    if risiko_rendah:
        prov_list_r = ", ".join([p.title() for p in risiko_rendah])
        st.success(f"""
        **🟢 Provinsi Risiko Rendah:**
        {prov_list_r}
    
        **Rekomendasi:**
        - Pertahankan program yang sudah berjalan baik
        - Sharing best practices ke provinsi lain
        - Lanjutkan monitoring untuk memastikan tren positif berlanjut
        """)

