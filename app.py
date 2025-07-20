# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Harga Minyak Mentah",
    page_icon="🛢️",
    layout="wide"
)

# --- FUNGSI-FUNGSI BANTUAN DENGAN CACHING ---
# Cache resource untuk model dan scaler agar tidak di-load ulang setiap interaksi
# app.py

# ... (kode lainnya tetap sama)

# --- FUNGSI-FUNGSI BANTUAN DENGAN CACHING ---
# Cache resource untuk model dan scaler agar tidak di-load ulang setiap interaksi
# app.py

@st.cache_resource
def load_model():
    """Memuat model LSTM-Attention dari file H5 dengan custom object."""
    custom_objects = {'Attention': tf.keras.layers.Attention}
    # PASTIKAN INI ADALAH NAMA FILE YANG BENAR DAN ADA DI FOLDER
    model = tf.keras.models.load_model(
        'model_kompatibel.h5', 
        custom_objects=custom_objects,
        compile=False
    )
    return model

@st.cache_resource
def load_scaler():
    """Memuat scaler dari file pkl."""
    scaler = joblib.load('scaler.pkl')
    return scaler

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
    
    # Ambil 30 hari terakhir dari data historis sebagai sekuens awal
    last_sequence = historical_data['price'].values[-lookback_period:]
    
    # Normalisasi sekuens awal
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    
    forecasted_values_scaled = []
    current_sequence = list(scaled_sequence)

    for _ in range(future_periods):
        # Reshape untuk input model: (1, 30, 1)
        input_array = np.array(current_sequence).reshape(1, lookback_period, 1)
        
        # Prediksi 1 langkah ke depan
        prediction_scaled = model.predict(input_array)[0]
        
        # Simpan hasil prediksi
        forecasted_values_scaled.append(prediction_scaled)
        
        # Perbarui sekuens: hapus data paling lama, tambahkan hasil prediksi
        current_sequence.pop(0)
        current_sequence.append(prediction_scaled)
        
    # Kembalikan ke skala asli
    forecasted_values = scaler.inverse_transform(forecasted_values_scaled)
    return forecasted_values.flatten()


# --- UI UTAMA STREAMLIT ---

st.title('🛢️ Dashboard Prediksi Harga Minyak Mentah (WTI)')
st.markdown("Dashboard ini menggunakan model **LSTM dengan Attention Mechanism** untuk memprediksi harga minyak mentah.")

# Memuat semua komponen
try:
    model = load_model()
    scaler = load_scaler()
    df = load_data('coba_data_harian_aligned.csv')
    LOOKBACK_PERIOD = 30 # Sesuai dengan window size di notebook
except FileNotFoundError:
    st.error("Pastikan file `coba_data_harian_aligned.csv`, `model_ANDA_SALAH.h5`, dan `scaler.pkl` berada di folder yang sama dengan `app.py`.")
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

# Metrik Evaluasi Model (dari notebook)
st.subheader("Performa Model pada Data Uji")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", "1.69")
col2.metric("MAE", "1.25")
col3.metric("MAPE", "1.81%")
st.markdown("*(Metrik dihitung pada 10% data terakhir sebagai data uji)*")

# Melakukan forecasting saat tombol ditekan
if st.sidebar.button('Mulai Prediksi', type="primary"):
    with st.spinner(f'Melakukan prediksi untuk {periods_to_forecast} hari ke depan...'):
        # Jalankan fungsi forecasting
        forecast = perform_forecasting(model, scaler, df, LOOKBACK_PERIOD, periods_to_forecast)
        
        # Buat tanggal untuk hasil prediksi
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods_to_forecast)
        
        # Gabungkan dalam DataFrame untuk ditampilkan
        forecast_df = pd.DataFrame({'Tanggal': forecast_dates, 'Harga Prediksi': forecast})

        st.subheader(f'Hasil Prediksi untuk {periods_to_forecast} Hari ke Depan')
        
        # --- PLOT VISUALISASI ---
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data historis (ambil 180 hari terakhir agar tidak terlalu padat)
        ax.plot(df.index[-180:], df['price'][-180:], label='Data Historis Aktual', color='royalblue')
        
        # Plot data prediksi
        ax.plot(forecast_dates, forecast, label='Data Prediksi', color='red', marker='o', linestyle='--')
        
        # Formatting plot
        ax.set_title('Prediksi Harga Minyak Mentah', fontsize=16)
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga (USD per Barel)', fontsize=12)
        ax.legend()
        ax.grid(True)
        fig.autofmt_xdate() # Memutar label tanggal agar tidak tumpang tindih
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        
        st.pyplot(fig)
        
        # Tampilkan data dalam tabel
        st.dataframe(forecast_df.set_index('Tanggal').style.format({"Harga Prediksi": "${:,.2f}"}))

else:
    st.info('Silakan atur jumlah hari di sidebar dan tekan tombol "Mulai Prediksi".')

# Menampilkan data mentah jika pengguna ingin melihatnya
with st.expander("Lihat Data Historis Mentah"):
    st.dataframe(df.tail(100).style.format({"price": "${:,.2f}", "kurs": "{:,.2f}", "GPRD": "{:,.2f}"}))