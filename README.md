# Proyek Prediksi Kelangsungan Hidup Titanic - Test Case Magang AI/ML

## Tentang Proyek Ini

Proyek ini adalah hasil dari test case magang AI/ML Developer yang saya ikuti. Tujuannya sederhana tapi menarik: membuat model Machine Learning untuk memprediksi siapa yang selamat dan tidak selamat di kapal Titanic. Lebih dari itu, model ini kemudian saya buatkan sebuah API (Antarmuka Pemrograman Aplikasi) supaya bisa dipakai oleh aplikasi lain.

**Apa saja yang saya kerjakan di proyek ini?**
1.  **Membangun Model ML:** Membuat "otak" cerdas yang bisa menebak kelangsungan hidup penumpang.
2.  **Menerapkan dalam Bentuk API:** Mengubah "otak" ini menjadi layanan yang bisa diakses oleh program lain, ibaratnya membuat tombol "prediksi" yang bisa dipakai siapa saja.
3.  **Dokumentasi dan Presentasi:** Merapikan semua pekerjaan dan mempersiapkan demo.

## Struktur Proyek

Saya mencoba mengatur folder proyek ini serapi mungkin agar mudah diikuti dan dipahami. Kira-kira seperti ini tampilannya:

## Struktur Proyek

Saya mencoba mengatur folder proyek ini serapi mungkin agar mudah diikuti dan dipahami. Kira-kira seperti ini tampilannya:

> titanic_survival_predictor/
> |-- app/
> |   `-- main.py
> |-- data/
> |   |-- raw/
> |   `-- processed/
> |-- models/
> |   |-- best_titanic_model.pkl
> |   |-- scaler.pkl
> |   `-- features_columns.pkl
> |-- notebooks/
> |   |-- 01_eda_preprocessing.ipynb
> |   `-- 02_model_training_evaluation.ipynb
> |-- src/
> |   `-- data_pipeline.py
> |-- .gitignore
> |-- README.md
> `-- requirements.txt

## Cara Menjalankan Proyek Ini

Untuk mencoba proyek ini di komputer Anda, ikuti langkah-langkah di bawah ini:

1.  **Kloning Repositori:**
    Buka terminal/Command Prompt Anda dan unduh kode proyek ini:
    ```bash
    git clone [https://github.com/seviamutiara/titanic-survival-prediction-ml-internship.git](https://github.com/seviamutiara/titanic-survival-prediction-ml-internship.git)
    cd titanic_survival_predictor
    ```

2.  **Siapkan Lingkungan Virtual (Virtual Environment):**
    Saya sangat merekomendasikan menggunakan lingkungan virtual agar pustaka proyek ini tidak bercampur dengan pustaka Python lain di sistem Anda.
    ```bash
    python -m venv venv_titanic
    ```
    * **Untuk pengguna Windows:**
        ```bash
        venv_titanic\Scripts\activate
        ```
    * **Untuk pengguna macOS/Linux:**
        ```bash
        source venv_titanic/bin/activate
        ```

3.  **Instal Pustaka yang Dibutuhkan:**
    Setelah lingkungan virtual aktif, instal semua pustaka Python yang tercantum di `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Mendapatkan Data

Dataset Titanic yang asli perlu Anda unduh sendiri dari Kaggle.

1.  **Unduh Dataset:**
    Kunjungi halaman kompetisi Titanic di Kaggle dan unduh file `train.csv` serta `test.csv`:
    [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)

2.  **Letakkan File Dataset:**
    Setelah diunduh, pindahkan kedua file tersebut (`train.csv` dan `test.csv`) ke dalam direktori:
    `titanic_survival_predictor/data/raw/`
    *(Catatan: File-file data mentah ini tidak saya sertakan di GitHub karena biasanya ukurannya besar dan untuk menjaga kerapian repositori.)*

## Menggunakan Proyek Ini

### 1. Mempelajari dan Melatih Model via Jupyter Notebooks

Notebook-notebook ini adalah "catatan" saya selama proses pengerjaan. Di sini Anda bisa melihat bagaimana saya menganalisis data, membersihkannya, membuat fitur-fitur baru, hingga melatih dan mengevaluasi modelnya.

1.  **Mulai JupyterLab:**
    Pastikan lingkungan virtual Anda aktif, lalu jalankan perintah ini dari direktori utama proyek:
    ```bash
    jupyter lab
    ```
2.  **Jelajahi Notebooks:**
    Di antarmuka JupyterLab yang muncul di browser, navigasikan ke folder `notebooks/`. Saya sarankan Anda membuka dan menjalankan file-file berikut secara berurutan:
    * `01_eda_preprocessing.ipynb`: Di sini saya melakukan Analisis Data Eksplorasi (EDA) dan semua langkah pra-pemrosesan data. Anda bisa melihat bagaimana saya menangani data yang hilang dan menyiapkan fitur.
    * `02_model_training_evaluation.ipynb`: Notebook ini berisi kode untuk melatih beberapa model Machine Learning, mengevaluasi kinerja masing-masing, dan menentukan model terbaik yang akan saya gunakan untuk API.
    * **Penting:** Pastikan untuk menjalankan semua *cell* di kedua notebook ini. Hasilnya akan menyimpan "alat" pra-pemrosesan dan model yang sudah jadi ke dalam direktori `models/`.

### 2. Menjalankan REST API

Setelah model saya terlatih dan semua "alat" pra-pemrosesan tersimpan, saya membuat sebuah API agar model ini bisa diakses.

1.  **Mulai Server API:**
    Buka **terminal/Command Prompt baru** (jangan tutup yang menjalankan JupyterLab jika masih ada). Pastikan lingkungan virtual Anda aktif.
    Navigasikan ke direktori `app/` dan jalankan skrip utama API:
    ```bash
    cd app
    python main.py
    ```
    Server API akan mulai berjalan, biasanya di alamat `http://127.0.0.1:5000`. Biarkan terminal ini tetap terbuka selama API Anda ingin diakses.

2.  **Menguji API:**
    Anda bisa menguji *endpoint* API menggunakan alat seperti [Postman](https://www.postman.com/downloads/) atau `cURL` (alat berbasis teks di terminal).

    #### A. Endpoint Cek Kesehatan (`/health`)
    * **Apa fungsinya?** Untuk memeriksa apakah API saya berjalan dengan baik dan modelnya sudah berhasil dimuat.
    * **Alamat:** `http://127.0.0.1:5000/health`
    * **Metode:** `GET` (Anda tinggal akses alamatnya di browser atau kirim permintaan GET di Postman tanpa body).
    * **Contoh Respon (JSON):**
        ```json
        {
            "status": "healthy",
            "message": "API is running and model loaded."
        }
        ```

    #### B. Endpoint Prediksi (`/predict`)
    * **Apa fungsinya?** Di sinilah Anda bisa mengirim data penumpang dan mendapatkan prediksi kelangsungan hidup.
    * **Alamat:** `http://127.0.0.1:5000/predict`
    * **Metode:** `POST`
    * **Header yang perlu ditambahkan:** `Content-Type: application/json`
    * **Contoh Data Input (JSON di bagian Body Postman, pilih `raw` dan `JSON`):**
        ```json
        [
            {
                "PassengerId": 1,
                "Pclass": 3,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 22,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": null,
                "Embarked": "S"
            },
            {
                "PassengerId": 2,
                "Pclass": 1,
                "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                "Sex": "female",
                "Age": 38,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "PC 17599",
                "Fare": 71.2833,
                "Cabin": "C85",
                "Embarked": "C"
            }
        ]
        ```
    * **Contoh Respon (JSON):**
        ```json
        [
            {
                "input_data": { /* data input yang sama */ },
                "predicted_survival": 0,  # 0 = Tidak Selamat, 1 = Selamat
                "probability_not_survived": 0.8543,
                "probability_survived": 0.1457
            },
            {
                "input_data": { /* data input yang sama */ },
                "predicted_survival": 1,
                "probability_not_survived": 0.0498,
                "probability_survived": 0.9502
            }
        ]
        ```

    #### C. Endpoint Latih Ulang (`/retrain`)
    * **Apa fungsinya?** Memungkinkan model dilatih ulang menggunakan dataset `train.csv` asli. Berguna jika ada pembaruan data atau ingin model belajar ulang.
    * **Alamat:** `http://127.0.0.1:5000/retrain`
    * **Metode:** `POST`
    * **Headers:** `Content-Type: application/json`
    * **Contoh Data Input (JSON kosong):**
        ```json
        {}
        ```
    * **Contoh Respon (JSON):**
        ```json
        {
            "status": "success",
            "message": "Model retrained successfully."
        }
        ```
        *(Anda bisa melihat pesan "Model re-trained and saved successfully." di terminal tempat API berjalan.)*

## Detail Model

* **Algoritma yang Digunakan:** Proyek ini mengeksplorasi tiga algoritma klasifikasi:
    * Logistic Regression
    * Random Forest Classifier
    * Gradient Boosting Classifier
* **Model Terbaik:**
    Setelah evaluasi dengan *cross-validation*, model **Gradient Boosting Classifier** dipilih sebagai model terbaik untuk deployment. Ini berdasarkan performa **F1 Score** (0.7770) dan juga metrik lain seperti Akurasi dan ROC AUC yang paling tinggi di antara model-model yang diuji.
* **Metrik Kinerja (Rata-rata Cross-Validation):**
    Berikut adalah rata-rata metrik kinerja untuk setiap model yang diuji:

    **Model: Logistic Regression**
    * Accuracy: 0.8249
    * Precision: 0.7895
    * Recall: 0.7425
    * F1 Score: 0.7641
    * ROC AUC: 0.8706

    **Model: Random Forest**
    * Accuracy: 0.8103
    * Precision: 0.7634
    * Recall: 0.7338
    * F1 Score: 0.7479
    * ROC AUC: 0.8749

    **Model: Gradient Boosting (Terbaik)**
    * Accuracy: 0.8350
    * Precision: 0.8102
    * Recall: 0.7485
    * F1 Score: 0.7770
    * ROC AUC: 0.8833

## Pembelajaran & Tantangan

Mengerjakan proyek ini dalam waktu yang terbatas sangat menantang tapi juga memberi banyak pelajaran berharga tentang alur kerja Machine Learning dari awal hingga akhir.

Beberapa hal yang saya pelajari dan tantangan yang saya hadapi:

* **Pentingnya Pra-pemrosesan Data:** Saya benar-benar menyadari betapa krusialnya membersihkan data yang hilang, membuat fitur baru (seperti 'Title' atau 'FamilySize'), dan mengubah tipe data agar model bisa belajar dengan baik.
* **Konsistensi adalah Kunci:** Memastikan semua langkah pra-pemrosesan yang saya terapkan saat melatih model juga diterapkan persis sama pada data baru yang masuk ke API. Ini saya atasi dengan membuat modul `data_pipeline.py` terpisah.
* **Menangani Peringatan (`Warnings`):** Saya sempat menemui beberapa peringatan dari pustaka seperti Pandas atau Seaborn. Meskipun bukan *error*, saya belajar untuk memahami maknanya dan memperbarui kode agar lebih *future-proof* dan bersih.
* **Membangun dan Menguji API:** Ini adalah pengalaman baru bagi saya dalam mengubah model ML menjadi layanan yang bisa diakses oleh aplikasi lain, dan saya belajar bagaimana mengujinya dengan Postman.

## Rencana Perbaikan di Masa Depan

Jika saya punya waktu lebih, beberapa area yang ingin saya tingkatkan dari proyek ini adalah:

* **Penyetelan Hyperparameter Lebih Lanjut:** Saya ingin mencoba *tuning* parameter model lebih dalam menggunakan teknik seperti Grid Search atau Randomized Search dengan *range* yang lebih luas untuk mendapatkan performa yang lebih optimal.
* **Eksplorasi Fitur Tambahan:** Mungkin ada informasi tersembunyi di kolom seperti 'Ticket' atau kombinasi fitur lain yang bisa saya gali.
* **Deployment ke Cloud:** Akan sangat menarik untuk mencoba menempatkan API ini di platform *cloud* seperti Heroku atau Google Cloud Run agar bisa diakses secara publik.

## Kontak

Nama Saya: Sevia Mutiara H
seviamutiara68@gmail.com
www.linkedin.com/in/sevia-mutiara-01a242206

