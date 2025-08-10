# 💸 Loan Default Prediction

**Dashboard interaktif untuk memprediksi apakah pelanggan akan mengalami default (gagal bayar) atau tidak.**  
Dibuat menggunakan **Streamlit**, **Plotly**, dan **scikit-learn**, aplikasi ini dapat digunakan untuk analisis data, pelatihan model, evaluasi performa, dan prediksi individual langsung di browser.

---

## 📊 Deskripsi Dataset

Dataset utama adalah `Loan_default.csv`, yang berisi informasi profil peminjam dan detail pinjaman.  
Setiap baris mewakili satu pelanggan dengan beberapa variabel:

**Contoh kolom:**
- `LoanID` → ID unik pinjaman (tidak digunakan untuk training)
- `Age` → Umur peminjam
- `Income` → Pendapatan tahunan
- `LoanAmount` → Jumlah pinjaman
- `CreditScore` → Skor kredit pelanggan
- `MonthsEmployed` → Lama bekerja (bulan)
- `NumCreditLines` → Jumlah jalur kredit
- `InterestRate` → Suku bunga
- `LoanTerm` → Tenor pinjaman (bulan)
- `DTIRatio` → Debt-to-Income ratio
- `Education`, `EmploymentType`, `MaritalStatus` → Informasi demografi
- `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner` → Informasi keuangan/pinjaman
- `Default` → **Target** (1 = Default, 0 = Tidak Default)

---

## 🎯 Tujuan Project

- **Memprediksi** apakah seorang pelanggan akan default atau tidak.
- **Memberikan insight visual** faktor-faktor yang mempengaruhi default.
- **Membuat aplikasi interaktif** yang memungkinkan:
  - Upload dataset sendiri
  - Pilih fitur dan target
  - Latih model langsung di web
  - Ubah *decision threshold*
  - Lihat metrik evaluasi dan visualisasi EDA
  - Prediksi untuk satu pelanggan

---

## 🔍 Tahapan Pengolahan Data

### 1. **Data Understanding**
- Memeriksa tipe data & missing values.
- Menentukan kolom target (`Default`) dan memisahkannya dari fitur.
- Menghapus kolom ID unik (`LoanID`) dari model.

### 2. **Data Preprocessing**
- **Numerik**: imputasi median, optional standardisasi.
- **Kategorikal**: imputasi modus, One-Hot Encoding.
- Mengabaikan kolom kategorikal dengan jumlah kategori terlalu besar (>20) pada EDA.

### 3. **EDA (Exploratory Data Analysis)**
- Distribusi default vs tidak default.
- Distribusi fitur numerik dengan highlight target.
- Proporsi kategori untuk fitur kategorikal.
- Semua visualisasi menggunakan **Plotly Dark Theme**.

4. **Modeling**  
   Model yang tersedia di aplikasi:
   - Logistic Regression
   - Random Forest
   - **XGBoost Classifier**
   
   **Hyperparameter XGBoost diatur via sidebar:**
   - `n_estimators`
   - `max_depth`
   - `learning_rate`
   - `subsample`
   - `colsample_bytree`
   - `scale_pos_weight` (opsi auto hitung dari data untuk menangani class imbalance)

   📌 *Kapan pakai `scale_pos_weight`?*  
   Gunakan ketika data target tidak seimbang (misal default hanya 10–20%). Opsi **Auto** akan menghitung nilai yang tepat dari rasio kelas mayoritas/minoritas di data training.

5. **Evaluation**  
   - Accuracy, Precision, Recall, F1, ROC AUC  
   - Confusion Matrix  
   - ROC Curve  
   - Feature Importance (Permutation)  

6. **Prediction**  
   - Form input untuk prediksi individual  
   - Menampilkan hasil + probabilitas default  
   - Menggunakan threshold yang sama seperti saat training  

---

🚀 **[Coba Aplikasinya di Sini](https://loandefaultprediction-luthfi.streamlit.app)**
