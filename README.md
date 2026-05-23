# ANFIS Hybrid Facial Emotion & Stress Analyzer
### Streamlit Demo App · Soft Computing Project

**Tim:** 
| Nama | NPM |
|-------|-----------|
| Siti Aisyah Nurdyanti | 140810230015 |
| Clarisya Adeline | 140810230017 |
| Nazwa Nashatasya | 140810230019 |

**Dataset:** FER2013 (Kaggle) — 35.887 gambar wajah grayscale 48×48 px

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Letakkan weights model (opsional)
Jika kamu sudah melatih model dari notebook, copy file weights ke folder yang sama:
```
anfis_emotion_model.weights.h5    ← dari /content/anfis_emotion_model.weights.h5
```
Jika file tidak ada, app akan berjalan dalam **mode simulasi** (demo preprocessing pipeline tetap berjalan penuh).

### 3. Jalankan app
```bash
streamlit run app.py
```
App akan terbuka di `http://localhost:8501`

---

## Fitur Aplikasi

| Fitur | Deskripsi |
|-------|-----------|
| Upload Gambar | Upload JPG/PNG wajah |
| Webcam Capture | Ambil foto langsung dari kamera |
| Preprocessing Pipeline | Identik dengan training: resize 48×48, CLAHE, normalize, LBP, HOG |
| Prediksi Emosi | 7 kelas (angry, disgust, fear, happy, neutral, sad, surprise) + confidence score |
| Bar Chart Probabilitas | Distribusi probabilitas semua 7 emosi |
| Stress Level Gauge | Gauge 0–100 dengan kategori Low / Moderate / High |
| LBP Map | Visualisasi tekstur kerutan (colormap: hot) |
| HOG Map | Visualisasi gradien kontur wajah (colormap: inferno) |
| LBP Histogram | Bar chart 64 dari 256 bin LBP |
| Disclaimer | Tampil wajib di halaman utama |

---

## Arsitektur Model

```
Input Image (48×48×1)
     │
     ├── CNN Branch (MobileNetV2) → 256-dim
     ├── LBP Branch (multi-radius) → 128-dim  
     └── HOG Branch (12 orientations) → 128-dim
                  │
         Concatenate (512-dim)
         Cross-Attention
         ANFIS Projection (64-dim)
                  │
         ┌─────────────────┐
         │   ANFIS CORE    │
         │  Dual Fuzzy MF  │  (Gaussian + Bell, 5 MF)
         │  48 Fuzzy Rules │  (T-norm product)
         │  Normalization  │
         │  TSK Consequent │  (einsum)
         │  LayerNorm+GELU │
         └─────────────────┘
                  │
         Residual + LayerNorm
                  │
     ┌────────────┴──────────┐
     │ Emotion (7, softmax)  │  Stress (1, sigmoid×100)
     └───────────────────────┘
```

---

## Disclaimer
> Sistem ini merupakan **prototipe akademik** dalam mata kuliah Soft Computing.  
> Output bukan diagnosis klinis. Tidak dimaksudkan sebagai pengganti penilaian profesional.
