# check_tf.py

import tensorflow as tf
import os

# Nonaktifkan pesan log TensorFlow yang tidak perlu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("--- Memulai Uji Isolasi ---")

# 1. Periksa Versi TensorFlow
try:
    print(f"Versi TensorFlow yang terinstal: {tf.__version__}")
except Exception as e:
    print(f"Gagal memeriksa versi TensorFlow: {e}")
    exit() # Keluar jika TensorFlow bahkan tidak bisa diimpor

# 2. Coba Muat Model
NAMA_MODEL_ANDA = 'model_kompatibel.h5' # <--- GANTI JIKA NAMA FILE ANDA BERBEDA
print(f"\nMencoba memuat model: {NAMA_MODEL_ANDA}")

if not os.path.exists(NAMA_MODEL_ANDA):
    print(f"!!! ERROR: File model '{NAMA_MODEL_ANDA}' tidak ditemukan di folder ini.")
    exit()

try:
    custom_objects = {'Attention': tf.keras.layers.Attention}
    model = tf.keras.models.load_model(
        NAMA_MODEL_ANDA,
        custom_objects=custom_objects,
        compile=False
    )
    print("\n✅ BERHASIL! Model sukses dimuat.")
    model.summary() # Tampilkan ringkasan model jika berhasil
except Exception as e:
    print("\n❌ GAGAL! Terjadi error saat memuat model.")
    print("--- Pesan Error Lengkap ---")
    # Cetak traceback error secara detail
    import traceback
    traceback.print_exc()
    print("--------------------------")

print("\n--- Uji Isolasi Selesai ---")