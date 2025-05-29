# src/data_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Global variables untuk menyimpan artefak yang akan dimuat saat modul diimpor ---
_scaler = None
_features_columns = None
_train_df_raw_for_context = None

# Fungsi untuk memuat artefak (scaler, features_columns, dan data training mentah)
def _load_artifacts_and_context():
    global _scaler, _features_columns, _train_df_raw_for_context
    try:
        # Perhatikan path relatifnya. Dari src/, untuk models/ dan data/raw/ kita naik satu level (..)
        _scaler = joblib.load(os.path.join(os.path.dirname(__file__), '../models/scaler.pkl'))
        _features_columns = joblib.load(os.path.join(os.path.dirname(__file__), '../models/features_columns.pkl'))
        print("Preprocessing artifacts loaded.")

        # Muat train_df_raw untuk konteks imputasi (median/mode)
        _train_df_raw_for_context = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/raw/train.csv'))
        print("Original train_df_raw loaded for preprocessing context.")

    except FileNotFoundError as e:
        print(f"Error loading artifact: {e}. Pastikan file .pkl dan data mentah ada di direktori 'models/' dan 'data/raw/' yang benar.")
        _scaler = None
        _features_columns = None
        _train_df_raw_for_context = None
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga saat memuat artefak: {e}")
        _scaler = None
        _features_columns = None
        _train_df_raw_for_context = None

# Panggil fungsi ini saat modul src/data_pipeline.py diimpor pertama kali
_load_artifacts_and_context()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan langkah-langkah pra-pemrosesan yang sama seperti saat pelatihan model.
    Mengasumsikan _scaler, _features_columns, dan _train_df_raw_for_context sudah dimuat secara global.
    """
    if _scaler is None or _features_columns is None or _train_df_raw_for_context is None:
        raise RuntimeError("Artefak pra-pemrosesan tidak dimuat. Tidak dapat memproses data.")

    df = df.copy() # Bekerja pada salinan dataframe agar tidak mengubah input asli

    # --- Penanganan Missing Values (Menggunakan konteks dari train_df_raw_for_context) ---
    if 'Age' in df.columns and df['Age'].isnull().any():
        df['Age'].fillna(_train_df_raw_for_context['Age'].median(), inplace=True)
    if 'Embarked' in df.columns and df['Embarked'].isnull().any():
        df['Embarked'].fillna(_train_df_raw_for_context['Embarked'].mode()[0], inplace=True)
    if 'Fare' in df.columns and df['Fare'].isnull().any():
        df['Fare'].fillna(_train_df_raw_for_context['Fare'].median(), inplace=True)


    # --- Feature Engineering ---
    # Title
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip() if pd.notnull(x) and isinstance(x, str) and ',' in x and '.' in x else 'Rare')
    title_mapping = {
        "Mme": "Mrs", "Mlle": "Miss", "Ms": "Miss", "Lady": "Mrs", "Countess": "Mrs",
        "Dona": "Mrs", "Sir": "Mr", "Don": "Mr", "Major": "Officer", "Col": "Officer",
        "Capt": "Officer", "Dr": "Officer", "Rev": "Officer", "Jonkheer": "Rare",
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master"
    }
    df['Title'] = df['Title'].replace(title_mapping)
    # Menangani gelar yang mungkin tidak ada di mapping (ubah jadi 'Rare')
    df['Title'] = df['Title'].apply(lambda x: x if x in title_mapping.values() else 'Rare')


    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Deck from Cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'U')

    # FarePerPerson
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerPerson'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['FarePerPerson'].fillna(df['FarePerPerson'].median(), inplace=True)


    # --- Hapus Kolom Asli ---
    df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId'], axis=1, inplace=True, errors='ignore')

    # --- One-Hot Encoding ---
    categorical_cols = ['Sex', 'Embarked', 'Pclass', 'Title', 'Deck']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str) # Pastikan tipe data string sebelum get_dummies

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Scaling Numerik ---
    numerical_cols_to_scale = ['Age', 'Fare', 'FamilySize', 'FarePerPerson']
    cols_to_scale_in_df = [col for col in numerical_cols_to_scale if col in df.columns]

    if cols_to_scale_in_df and _scaler:
        df[cols_to_scale_in_df] = _scaler.transform(df[cols_to_scale_in_df])


    # --- Penjajaran Kolom (SANGAT KRITIS UNTUK API) ---
    # Pastikan semua kolom yang diharapkan model ada, dan dalam urutan yang benar
    for col in _features_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[_features_columns] # Pastikan urutan kolom sama persis

    return df