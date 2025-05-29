# src/data_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Deklarasi variabel global untuk menyimpan artefak pra-pemrosesan.
# Variabel-variabel ini akan dimuat saat modul diimpor pertama kali.
_scaler = None
_features_columns = None
_train_df_raw_for_context = None

def _load_artifacts_and_context():
    """
    Memuat artefak penting (scaler, daftar fitur, dan data pelatihan mentah)
    yang diperlukan untuk pra-pemrosesan data secara konsisten.
    """
    global _scaler, _features_columns, _train_df_raw_for_context
    try:
        # Menentukan path relatif untuk file .pkl dan data mentah.
        # os.path.dirname(__file__) mendapatkan direktori skrip saat ini (src/).
        # '..' berarti naik satu level ke direktori root proyek.
        _scaler = joblib.load(os.path.join(os.path.dirname(__file__), '../models/scaler.pkl'))
        _features_columns = joblib.load(os.path.join(os.path.dirname(__file__), '../models/features_columns.pkl'))
        print("Artefak pra-pemrosesan berhasil dimuat.")

        # Memuat data pelatihan mentah untuk tujuan konteks imputasi (median/modus).
        _train_df_raw_for_context = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/raw/train.csv'))
        print("Data pelatihan mentah asli dimuat untuk konteks pra-pemrosesan.")

    except FileNotFoundError as e:
        print(f"Error memuat artefak: {e}. Pastikan file .pkl dan data mentah berada di direktori 'models/' dan 'data/raw/' yang benar.")
        _scaler = None
        _features_columns = None
        _train_df_raw_for_context = None
    except Exception as e:
        print(f"Terjadi kesalahan tidak terduga saat memuat artefak: {e}")
        _scaler = None
        _features_columns = None
        _train_df_raw_for_context = None

# Memanggil fungsi pemuatan artefak saat modul ini pertama kali diimpor.
_load_artifacts_and_context()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan langkah-langkah pra-pemrosesan data yang sama seperti yang digunakan
    selama pelatihan model.

    Args:
        df (pd.DataFrame): DataFrame input mentah yang akan diproses.

    Returns:
        pd.DataFrame: DataFrame yang telah diproses dan siap untuk inferensi model.
    """
    # Memastikan artefak yang diperlukan sudah dimuat sebelum pra-pemrosesan.
    if _scaler is None or _features_columns is None or _train_df_raw_for_context is None:
        raise RuntimeError("Artefak pra-pemrosesan tidak termuat. Tidak dapat memproses data.")

    # Membuat salinan DataFrame untuk menghindari modifikasi input asli.
    df = df.copy()

    # --- Penanganan Missing Values (Imputasi) ---
    # Mengisi nilai yang hilang menggunakan median/modus dari data pelatihan asli.
    if 'Age' in df.columns and df['Age'].isnull().any():
        df['Age'] = df['Age'].fillna(_train_df_raw_for_context['Age'].median())
    if 'Embarked' in df.columns and df['Embarked'].isnull().any():
        df['Embarked'] = df['Embarked'].fillna(_train_df_raw_for_context['Embarked'].mode()[0])
    if 'Fare' in df.columns and df['Fare'].isnull().any():
        df['Fare'] = df['Fare'].fillna(_train_df_raw_for_context['Fare'].median())


    # --- Rekayasa Fitur (Feature Engineering) ---
    # Ekstraksi 'Title' dari kolom 'Name'.
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip() if pd.notnull(x) and isinstance(x, str) and ',' in x and '.' in x else 'Rare')
    title_mapping = {
        "Mme": "Mrs", "Mlle": "Miss", "Ms": "Miss", "Lady": "Mrs", "Countess": "Mrs",
        "Dona": "Mrs", "Sir": "Mr", "Don": "Mr", "Major": "Officer", "Col": "Officer",
        "Capt": "Officer", "Dr": "Officer", "Rev": "Officer", "Jonkheer": "Rare",
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master"
    }
    df['Title'] = df['Title'].replace(title_mapping)
    # Menangani gelar yang tidak terpetakan dalam mapping.
    df['Title'] = df['Title'].apply(lambda x: x if x in title_mapping.values() else 'Rare')


    # Menghitung 'FamilySize' (jumlah anggota keluarga termasuk penumpang itu sendiri).
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Membuat fitur 'IsAlone' (1 jika bepergian sendiri, 0 jika tidak).
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Mengekstrak 'Deck' dari 'Cabin' (huruf pertama atau 'U' untuk tidak diketahui).
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'U')

    # Menghitung 'FarePerPerson' (biaya tiket per anggota keluarga).
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    # Menangani nilai tak terhingga (inf) atau NaN yang mungkin muncul dari pembagian.
    df['FarePerPerson'] = df['FarePerPerson'].replace([np.inf, -np.inf], np.nan)
    df['FarePerPerson'] = df['FarePerPerson'].fillna(df['FarePerPerson'].median())


    # --- Penghapusan Kolom Asli ---
    # Menghapus kolom asli yang tidak lagi diperlukan setelah feature engineering.
    # 'errors=ignore' akan mencegah error jika kolom sudah tidak ada.
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId'], axis=1, inplace=True, errors='ignore')

    # --- One-Hot Encoding ---
    # Mengonversi fitur kategorikal ke representasi numerik (0 atau 1).
    categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Title', 'Deck']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str) # Memastikan tipe data string sebelum One-Hot Encoding

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Scaling Numerik ---
    # Menyamakan skala fitur numerik menggunakan StandardScaler yang sudah di-fit.
    numerical_cols_to_scale = ['Age', 'Fare', 'FamilySize', 'FarePerPerson']
    # Filter kolom numerik yang benar-benar ada di DataFrame input saat ini.
    cols_to_scale_in_df = [col for col in numerical_cols_to_scale if col in df.columns]

    if cols_to_scale_in_df and _scaler:
        df[cols_to_scale_in_df] = _scaler.transform(df[cols_to_scale_in_df])


    # --- Penjajaran Kolom (Kritis untuk Konsistensi Model) ---
    # Memastikan DataFrame output memiliki kolom yang sama persis dan dalam urutan yang sama
    # seperti data pelatihan model. Menambahkan kolom yang hilang (dengan nilai 0) dan
    # menghapus kolom ekstra.
    for col in _features_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[_features_columns] # Memastikan urutan kolom yang tepat

    return df