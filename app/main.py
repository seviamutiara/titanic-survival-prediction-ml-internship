# app/main.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os

# Tambahkan path ke direktori 'src' dan root proyek ke sys.path
# Ini memungkinkan kita mengimpor modul dari 'src' dan mengakses 'models/'/'data/'
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))) # Tambahkan root proyek

# Ini adalah cara yang lebih aman untuk impor modul lokal dan akses file
# Pastikan struktur folder Anda:
# - titanic_survival_predictor/
#   - app/main.py
#   - src/data_pipeline.py
#   - models/best_titanic_model.pkl
#   - data/raw/train.csv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import fungsi preprocessing dari src/data_pipeline.py
# Data_pipeline akan memuat artefak saat diimpor
from src.data_pipeline import preprocess_data

app = Flask(__name__)

# Global variable to store the loaded model
model = None

# Path ke data training asli (untuk endpoint retrain)
TRAIN_DATA_PATH = os.path.join(project_root, 'data', 'raw', 'train.csv')
MODEL_SAVE_PATH = os.path.join(project_root, 'models', 'best_titanic_model.pkl')


def load_model():
    """Loads the pre-trained model."""
    global model
    try:
        model = joblib.load(MODEL_SAVE_PATH)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Please ensure it's in the 'models/' directory.")
        model = None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        model = None

# Muat model saat aplikasi dimulai
with app.app_context():
     load_model()


# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running and model is loaded."""
    if model:
        return jsonify({"status": "healthy", "message": "API is running and model loaded."}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "API is running but model not loaded. Check server logs."}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions on new data."""
    if model is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 503

    try:
        # Dapatkan data input dari request JSON
        data = request.get_json(force=True)
        if not isinstance(data, list): # Mendukung prediksi single atau batch
            data = [data]

        # Konversi input JSON ke DataFrame
        input_df = pd.DataFrame(data)

        # Preprocessing data menggunakan fungsi dari data_pipeline.py
        processed_input = preprocess_data(input_df)

        # Lakukan prediksi
        predictions = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input).tolist() # Probabilitas selamat/tidak

        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "input_data": data[i], # Tampilkan input asli (opsional, untuk debugging/trace)
                "predicted_survival": int(pred), # 0 = Tidak Selamat, 1 = Selamat
                "probability_not_survived": round(prediction_proba[i][0], 4),
                "probability_survived": round(prediction_proba[i][1], 4)
            })

        return jsonify(results), 200

    except RuntimeError as re:
        return jsonify({"error": str(re), "message": "Preprocessing artifacts missing or corrupted. Detail: " + str(re) }), 500
    except Exception as e:
        print(f"Prediction error: {e}") # Log error untuk debugging
        return jsonify({"error": str(e), "message": "An error occurred during prediction. Check input data format and server logs."}), 400

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Endpoint to retrain the model with the original training data."""
    global model

    try:
        # Muat ulang data training asli
        if not os.path.exists(TRAIN_DATA_PATH):
            return jsonify({"error": f"Training data not found at {TRAIN_DATA_PATH}. Cannot retrain."}), 500

        train_df_retrain_raw = pd.read_csv(TRAIN_DATA_PATH)
        y_train_retrain = train_df_retrain_raw['Survived']
        X_train_retrain_raw = train_df_retrain_raw.drop('Survived', axis=1)

        print(f"Retraining: Loaded training data with shape {X_train_retrain_raw.shape}")

        # Preprocessing data training untuk retraining menggunakan fungsi data_pipeline.py
        # Perhatikan: fungsi preprocess_data akan menggunakan scaler dan features_columns
        # yang sudah dimuat. Jika Anda ingin melakukan *full* retraining (termasuk re-fitting scaler),
        # Anda perlu logika lebih lanjut di sini atau di src/model_training.py
        X_train_processed_for_retrain = preprocess_data(X_train_retrain_raw)

        # Latih ulang model (menggunakan jenis model yang sama dari model yang sudah dimuat)
        # Pastikan jenis modelnya sesuai, misal RandomForestClassifier()
        if model is None:
            # Fallback jika model belum dimuat, bisa pakai default atau model yang biasa
            from sklearn.ensemble import RandomForestClassifier
            retrained_model = RandomForestClassifier(random_state=42)
        else:
            retrained_model = type(model)() # Membuat instance baru dari jenis model yang sama

        retrained_model.fit(X_train_processed_for_retrain, y_train_retrain)

        # Perbarui model global
        model = retrained_model

        # Simpan model yang baru dilatih
        joblib.dump(model, MODEL_SAVE_PATH)
        print("Model re-trained and saved successfully.")

        return jsonify({"status": "success", "message": "Model retrained successfully."}), 200

    except RuntimeError as re:
        return jsonify({"error": str(re), "message": "Preprocessing artifacts missing or corrupted during retraining."}), 500
    except Exception as e:
        print(f"Retrain error: {e}") # Log error
        return jsonify({"error": str(e), "message": f"Gagal melatih ulang model. Detail: {str(e)}"}), 500


if __name__ == '__main__':
    # Jalankan aplikasi Flask di development mode
    # debug=True akan memberikan pesan error yang lebih detail dan reload otomatis
    app.run(host='0.0.0.0', port=5000, debug=True)