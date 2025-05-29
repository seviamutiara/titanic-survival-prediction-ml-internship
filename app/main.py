from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os

# Menambahkan root proyek ke sys.path untuk memungkinkan impor modul lokal
# dan akses ke direktori 'models/' dan 'data/'.
# Ini memastikan modul dan file dapat ditemukan terlepas dari di mana API dijalankan.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mengimpor fungsi pra-pemrosesan data dari modul kustom 'src.data_pipeline'.
# Modul ini akan memuat artefak pra-pemrosesan (scaler, features_columns) saat diimpor.
from src.data_pipeline import preprocess_data

app = Flask(__name__)

# Variabel global untuk menyimpan objek model ML yang dimuat.
model = None

# Mendefinisikan path ke dataset pelatihan mentah dan lokasi penyimpanan model.
TRAIN_DATA_PATH = os.path.join(project_root, 'data', 'raw', 'train.csv')
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'best_titanic_model.pkl')


def load_model():
    """Memuat model Machine Learning yang telah dilatih sebelumnya."""
    global model
    try:
        model = joblib.load(MODEL_SAVE_PATH)
        print("Model berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File model tidak ditemukan di {MODEL_SAVE_PATH}. Pastikan berada di direktori 'models/'.")
        model = None
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat model: {e}")
        model = None

# Memuat model saat aplikasi Flask pertama kali diinisialisasi.
with app.app_context():
    load_model()


# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint untuk memeriksa status API dan ketersediaan model.
    Mengembalikan status 'healthy' jika model termuat.
    """
    if model:
        return jsonify({"status": "healthy", "message": "API berjalan dan model termuat."}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "API berjalan tetapi model tidak termuat. Periksa log server."}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi berdasarkan data input baru.
    Menerima data penumpang dalam format JSON.
    """
    if model is None:
        return jsonify({"error": "Model tidak termuat. Tidak dapat melakukan prediksi."}), 503

    try:
        # Mendapatkan data input dari body request JSON.
        data = request.get_json(force=True)
        # Mendukung input tunggal atau batch (list of dictionaries).
        if not isinstance(data, list):
            data = [data]

        # Mengonversi input JSON menjadi Pandas DataFrame.
        input_df = pd.DataFrame(data)

        # Melakukan pra-pemrosesan data menggunakan fungsi dari data_pipeline.py.
        processed_input = preprocess_data(input_df)

        # Melakukan prediksi menggunakan model yang dimuat.
        predictions = model.predict(processed_input)
        # Mendapatkan probabilitas prediksi untuk setiap kelas (selamat/tidak selamat).
        prediction_proba = model.predict_proba(processed_input).tolist()

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "input_data": data[i], # Menyertakan input asli dalam respons (opsional, untuk debugging)
                "predicted_survival": int(pred), # Hasil prediksi (0 = Tidak Selamat, 1 = Selamat)
                "probability_not_survived": round(prediction_proba[i][0], 4),
                "probability_survived": round(prediction_proba[i][1], 4)
            })

        return jsonify(results), 200

    except RuntimeError as re:
        # Menangani error jika artefak pra-pemrosesan tidak dimuat atau rusak.
        return jsonify({"error": str(re), "message": "Artefak pra-pemrosesan hilang atau rusak. Detail: " + str(re) }), 500
    except Exception as e:
        # Menangani error umum selama proses prediksi.
        print(f"Error prediksi: {e}") # Log error untuk debugging server
        return jsonify({"error": str(e), "message": "Terjadi kesalahan saat prediksi. Periksa format data input dan log server."}), 400

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Endpoint untuk melatih ulang model menggunakan dataset pelatihan asli.
    Model yang baru dilatih akan disimpan kembali.
    """
    global model

    try:
        # Memuat ulang data pelatihan mentah dari file CSV.
        if not os.path.exists(TRAIN_DATA_PATH):
            return jsonify({"error": f"Data pelatihan tidak ditemukan di {TRAIN_DATA_PATH}. Tidak dapat melatih ulang."}), 500

        train_df_retrain_raw = pd.read_csv(TRAIN_DATA_PATH)
        y_train_retrain = train_df_retrain_raw['Survived']
        X_train_retrain_raw = train_df_retrain_raw.drop('Survived', axis=1)

        print(f"Pelatihan ulang: Data pelatihan dimuat dengan bentuk {X_train_retrain_raw.shape}")

        # Melakukan pra-pemrosesan data pelatihan mentah untuk proses retraining.
        # Catatan: Fungsi preprocess_data akan menggunakan scaler dan features_columns
        # yang sudah dimuat. Untuk retraining penuh (termasuk re-fitting scaler),
        # logika lebih lanjut diperlukan di sini atau di src/model_training.py.
        X_train_processed_for_retrain = preprocess_data(X_train_retrain_raw)

        # Melatih ulang model. Menggunakan tipe model yang sama dengan yang sudah dimuat.
        if model is None:
            # Fallback jika model belum dimuat; menggunakan RandomForestClassifier sebagai default.
            from sklearn.ensemble import RandomForestClassifier
            retrained_model = RandomForestClassifier(random_state=42)
        else:
            retrained_model = type(model)() # Membuat instance baru dari kelas model yang sama

        retrained_model.fit(X_train_processed_for_retrain, y_train_retrain)

        # Memperbarui model global dengan model yang baru dilatih.
        model = retrained_model

        # Menyimpan model yang baru dilatih ke lokasi penyimpanan model.
        joblib.dump(model, MODEL_SAVE_PATH)
        print("Model berhasil dilatih ulang dan disimpan.")

        return jsonify({"status": "success", "message": "Model berhasil dilatih ulang."}), 200

    except RuntimeError as re:
        # Menangani error jika artefak pra-pemrosesan bermasalah selama retraining.
        return jsonify({"error": str(re), "message": "Artefak pra-pemrosesan hilang atau rusak selama pelatihan ulang."}), 500
    except Exception as e:
        # Menangani error umum selama proses retraining.
        print(f"Error pelatihan ulang: {e}") # Log error untuk debugging server
        return jsonify({"error": str(e), "message": f"Gagal melatih ulang model. Detail: {str(e)}"}), 500


if __name__ == '__main__':
    # Menjalankan aplikasi Flask dalam mode pengembangan.
    # debug=True memberikan detail error yang lebih lengkap dan fitur auto-reload.
    app.run(host='0.0.0.0', port=5000, debug=True)