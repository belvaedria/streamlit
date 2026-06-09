import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

st.set_page_config(
    page_title="USA Apartment Rent Prediction",
    page_icon="🏙️",
    layout="wide"
)

# --- 1. MEMUAT FILE SCALER & DAFTAR KOLOM ASLI ---
@st.cache_resource
def load_preprocessing_files():
    try:
        scaler = joblib.load('robust_scaler.joblib')
        all_features = joblib.load('kolom_fitur.joblib')
        return scaler, all_features
    except Exception as e:
        st.error(f"Gagal memuat file preprocessing (scaler/kolom_fitur): {e}")
        st.stop()

scaler, ALL_FEATURES = load_preprocessing_files()


# --- 2. FUNGSI OTOMATIS PEMBACA MODEL ---
def get_model_files():
    model_folder = "model"
    joblib_files = glob.glob(os.path.join(model_folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(model_folder, "*.pkl"))
    return joblib_files + pkl_files

# @st.cache_resource 
def load_model(model_path):
    return joblib.load(model_path)


# --- 3. ANTARMUKA UTAMA WEB ---
st.title("🏙️ Apartments in USA Price Prediction App")
st.write(
    "Aplikasi ini digunakan untuk memprediksi estimasi harga sewa apartemen bulanan di USA "
    "berdasarkan spesifikasi bangunan dan lokasi secara manual."
)

# --- PILIH MODEL ---
st.header("Pilih Model")
model_files = get_model_files()

if len(model_files) == 0:
    st.error("Belum ada file model di folder `model/`.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]
selected_model_name = st.selectbox("Pilih model machine learning untuk prediksi:", model_names)
selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.success(f"Model aktif: {selected_model_name}")
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()


# --- 4. FORM INPUT DATA MANUAL ---
st.header("Input Spesifikasi Apartemen")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📐 Spesifikasi Fisik")
    square_feet = st.number_input("Luas Bangunan (Square Feet):", min_value=100, max_value=10000, value=800, step=10)
    bedrooms = st.number_input("Jumlah Kamar Tidur (Bedrooms):", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
    bathrooms = st.number_input("Jumlah Kamar Mandi (Bathrooms):", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
    
    has_photo_pilihan = st.radio("Apakah Iklan Memiliki Foto?", ("Ya", "Tidak"))
    has_photo = 1 if has_photo_pilihan == "Ya" else 0

with col2:
    st.subheader("📍 Kategori & Lokasi")
    
    categories = ["Apartment", "Home", "Short Term"]
    
    states = sorted([col.replace('state_', '') for col in ALL_FEATURES if col.startswith('state_')])
    cities = sorted([col.replace('cityname_', '') for col in ALL_FEATURES if col.startswith('cityname_')])
    if 'Others' in cities:
        cities.remove('Others')
    cities.append('Others')

    category_pilihan = st.selectbox("Tipe Kategori Properti:", categories)
    state_pilihan = st.selectbox("Negara Bagian (State):", states)
    city_pilihan = st.selectbox("Nama Kota (City Name):", cities)

    # Disederhanakan value pilihan dropdown-nya agar teksnya singkron saat dicari kuncinya
    pets_pilihan = st.selectbox("Kebijakan Hewan Peliharaan:", ("Cats,Dogs", "Dogs", "None"))


# --- 5. MENYUSUN DATA INPUT MENJADI KOLOM ASLI MODEL (VERSI FIX MURNI /) ---
input_row = {fitur: 0 for fitur in ALL_FEATURES}

input_row['square_feet'] = square_feet
input_row['bedrooms'] = bedrooms
input_row['bathrooms'] = bathrooms
input_row['has_photo'] = has_photo

# Logika mapping yang BENAR, menggunakan tanda slash '/' sesuai file joblib kamu
if category_pilihan != "Apartment":
    # Mengubah 'Home' -> 'category_housing/rent/home'
    key_cat = f"category_housing/rent/{category_pilihan.lower().replace(' ', '_')}"
    if key_cat in input_row:
        input_row[key_cat] = 1

key_state = f"state_{state_pilihan}"
if key_state in input_row:
    input_row[key_state] = 1

key_city = f"cityname_{city_pilihan}"
if key_city in input_row:
    input_row[key_city] = 1

key_pets = f"pets_allowed_{pets_pilihan}"
if key_pets in input_row:
    input_row[key_pets] = 1

# Ubah menjadi DataFrame dan kunci urutan kolomnya berdasarkan ALL_FEATURES
input_df = pd.DataFrame([input_row])
input_df = input_df[ALL_FEATURES]


# --- 6. PROSES PREDIKSI DAN PENAMPILAN HASIL SEWA ---
st.subheader("Data Input")
st.dataframe(input_df)

if st.button("Hitung Estimasi Harga Sewa"):
    try:
        # 1. Duplikat data input mentah
        input_scaled = input_df.copy()
        
        # 2. Lakukan scaling pada kolom numerik
        kolom_numerik = ['square_feet', 'bedrooms', 'bathrooms']
        input_scaled[kolom_numerik] = scaler.transform(input_scaled[kolom_numerik])
        
        # 3. Amankan kembali urutan kolom sesaat sebelum predict
        input_scaled = input_scaled[ALL_FEATURES]
        
        # 4. Prediksi nominal harganya
        prediction = model.predict(input_scaled)[0]
        
        st.header("💰 Hasil Prediksi")
        st.success(f"Estimasi Harga Sewa Properti: **${prediction:,.2f} / Bulan**")
        st.info(f"Dihitung secara real-time menggunakan algoritma: {selected_model_name}")
        
    except Exception as e:
        st.error(f"Terjadi error saat memproses prediksi: {e}")