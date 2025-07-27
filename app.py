# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from huggingface_hub import hf_hub_download 

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Harga Minyak Mentah",
    page_icon="🛢️",
    layout="wide"
)

# --- FUNGSI-FUNGSI BANTUAN DENGAN CACHING ---

@st.cache_resource
def load_model():
    """
    Mengunduh dan memuat model LSTM-Attention dari Hugging Face Hub.
    """
    # --- GANTI INI DENGAN REPO-ID KAMU ---
    REPO_ID = "moreno03/LSTM-LuongAttention"
    MODEL_FILENAME = "model_kompatibel.h5"
    # ------------------------------------

    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        custom_objects = {'Attention': tf.keras.layers.Attention}
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error saat mengunduh/memuat model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """
    Mengunduh dan memuat scaler dari Hugging Face Hub.
    """
    # --- GANTI INI DENGAN REPO-ID KAMU ---
    REPO_ID = "moreno03/LSTM-LuongAttention"
    SCALER_FILENAME = "scaler.pkl"
    # ------------------------------------
    
    try:
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename=SCALER_FILENAME)
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error saat mengunduh/memuat scaler: {e}")
        return None

# Cache data agar tidak dibaca dari CSV berulang kali
@st.cache_data
def load_data(filepath):
    """Memuat dan memproses data historis dari CSV."""
    df = pd.read_csv(filepath)
    df['waktu'] = pd.to_datetime(df['waktu'], format='%d/%m/%Y')
    df.set_index('waktu', inplace=True)
    return df

def perform_forecasting(model, scaler, historical_data, lookback_period, future_periods):
    """Melakukan forecasting autoregressive untuk beberapa periode ke depan."""
    last_sequence = historical_data['price'].values[-lookback_period:]
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    
    forecasted_values_scaled = []
    current_sequence = list(scaled_sequence)

    for _ in range(future_periods):
        input_array = np.array(current_sequence).reshape(1, lookback_period, 1)
        prediction_scaled = model.predict(input_array)[0]
        forecasted_values_scaled.append(prediction_scaled)
        current_sequence.pop(0)
        current_sequence.append(prediction_scaled)
        
    forecasted_values = scaler.inverse_transform(forecasted_values_scaled)
    return forecasted_values.flatten()


# --- UI UTAMA STREAMLIT ---

st.title('🛢️ Dashboard Prediksi Harga Minyak Mentah (WTI)')
st.markdown("Dashboard ini menggunakan model **LSTM dengan Attention Mechanism** untuk memprediksi harga minyak mentah.")

# Memuat semua komponen
# Pesan error disesuaikan untuk loading dari Hugging Face
model = load_model()
scaler = load_scaler()
try:
    df = load_data('coba_data_harian_aligned.csv')
    LOOKBACK_PERIOD = 30
except FileNotFoundError:
    st.error("File data `coba_data_harian_aligned.csv` tidak ditemukan. Pastikan file ini ada di repositori GitHub kamu.")
    st.stop()

# Hentikan aplikasi jika model atau scaler gagal dimuat
if not model or not scaler:
    st.error("Model atau Scaler gagal dimuat dari Hugging Face Hub. Pastikan REPO_ID dan nama file sudah benar, dan repo bersifat publik.")
    st.stop()


# --- SIDEBAR UNTUK INPUT PENGGUNA ---
st.sidebar.header("Pengaturan Prediksi")
periods_to_forecast = st.sidebar.slider(
    'Pilih jumlah hari untuk diprediksi:',
    min_value=1,
    max_value=90,
    value=30,
    step=1
)

# --- TAMPILAN UTAMA ---
st.subheader("Performa Model pada Data Uji")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", "1.69")
col2.metric("MAE", "1.25")
col3.metric("MAPE", "1.81%")
st.markdown("*(Metrik dihitung pada 10% data terakhir sebagai data uji)*")

if st.sidebar.button('Mulai Prediksi', type="primary"):
    with st.spinner(f'Melakukan prediksi untuk {periods_to_forecast} hari ke depan...'):
        forecast = perform_forecasting(model, scaler, df, LOOKBACK_PERIOD, periods_to_forecast)
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_to_forecast)
        forecast_df = pd.DataFrame({'Tanggal': forecast_dates, 'Harga Prediksi': forecast})

        st.subheader(f'Hasil Prediksi untuk {periods_to_forecast} Hari ke Depan')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-180:], df['price'][-180:], label='Data Historis Aktual', color='royalblue')
        ax.plot(forecast_dates, forecast, label='Data Prediksi', color='red', marker='o', linestyle='--')
        
        ax.set_title('Prediksi Harga Minyak Mentah', fontsize=16)
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga (USD per Barel)', fontsize=12)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        
        st.pyplot(fig)
        st.dataframe(forecast_df.set_index('Tanggal').style.format({"Harga Prediksi": "${:,.2f}"}))
else:
    st.info('Silakan atur jumlah hari di sidebar dan tekan tombol "Mulai Prediksi".')

with st.expander("Lihat Data Historis Mentah"):
    st.dataframe(df.tail(100).style.format({"price": "${:,.2f}", "kurs": "{:,.2f}", "GPRD": "{:,.2f}"}))