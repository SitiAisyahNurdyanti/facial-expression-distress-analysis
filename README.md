# ANFIS Hybrid Facial Emotion & Stress Analyzer
### Streamlit Demo App · Soft Computing Project

**Tim:**
| Nama | NPM |
|------|-----|
| Siti Aisyah Nurdyanti (Isya) | 140810230015 |
| Clarisya Adeline (Ica) | 140810230017 |
| Nazwa Nashatasya (Awa) | 140810230019 |

**Dataset:** FER2013 (Kaggle) — 35.887 gambar wajah grayscale 48×48 px

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan file wajib sebelum menjalankan app

App membutuhkan dua file model deteksi wajah OpenCV DNN. **App tidak akan jalan tanpa file ini.**

```
deploy.prototxt                           ← konfigurasi arsitektur Caffe
res10_300x300_ssd_iter_140000.caffemodel  ← bobot model deteksi wajah
```

Keduanya bisa diunduh dari repositori OpenCV:
- `deploy.prototxt` → [link](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
- `.caffemodel` → [link](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

Letakkan keduanya di folder yang sama dengan `app.py`.

### 3. Letakkan weights ANFIS (opsional)

```
anfis_emotion_model.weights.h5    ← dari /content/anfis_emotion_model.weights.h5
```

Jika file tidak ada, app tetap berjalan dalam **mode simulasi** — preprocessing pipeline dan seluruh UI tetap aktif penuh.

### 4. Jalankan app

```bash
streamlit run app.py
```

App akan terbuka di `http://localhost:8501`

---

## Fitur Aplikasi

| Fitur | Deskripsi |
|-------|-----------|
| **Upload Gambar** | Upload JPG/PNG/JPEG; otomatis crop wajah via DNN |
| **Capture Webcam** | Ambil foto langsung dari kamera (`st.camera_input`); analisis penuh dengan grafik |
| **Real-Time Webcam** | Stream langsung via `streamlit-webrtc`; prediksi frame-by-frame tanpa ambil foto |
| **Face Detection** | OpenCV DNN (ResNet SSD) — threshold confidence 0.5 untuk upload, fallback square crop |
| **Preprocessing Pipeline** | Identik dengan training: grayscale → resize 48×48 → CLAHE (clipLimit=2.0, tile 4×4) → normalize /255 |
| **Prediksi Emosi** | 7 kelas: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise + confidence score |
| **Top 3 Prediksi** | Ranking 3 emosi teratas dengan probabilitas masing-masing |
| **Stress Score** | Skor 0–100 berdasarkan emotion-to-stress weights; kategori Low / Moderate / High |
| **Bar Chart Probabilitas** | Horizontal bar chart distribusi probabilitas 7 emosi |
| **Stress Gauge** | Gauge semicircle 0–100 dengan segmen warna |
| **LBP Map** | Visualisasi tekstur kerutan wajah (colormap: hot) |
| **HOG Map** | Visualisasi gradien kontur wajah (colormap: inferno) |
| **LBP Histogram** | Bar chart 64 dari 256 bin LBP |
| **Tabel Dimensi Fitur** | Ringkasan branch CNN/LBP/HOG dengan dimensi dan metode |
| **Arsitektur Detail** | Expander pipeline lengkap dari input hingga output head |
| **Mode Simulasi** | Fallback deterministik berbasis LBP seed jika weights tidak tersedia |
| **Disclaimer** | Tampil wajib di halaman utama setiap saat |

### Perbedaan mode input

| | Upload / Capture | Real-Time Webcam |
|---|---|---|
| Trigger prediksi | Manual (setelah gambar dipilih) | Otomatis tiap 5 frame |
| Grafik & visualisasi | ✅ Lengkap (bar, gauge, LBP, HOG) | ❌ Tidak ditampilkan (hemat performa) |
| Overlay hasil | Kartu UI Streamlit | Text overlay langsung di video frame |
| Bounding box | Tidak ditampilkan | ✅ Warna sesuai stress level |

---

## Arsitektur Model

```
Input Image (48×48×1)
      │
      ├──[CNN Branch]── Resize(96×96) → MobileNetV2 (pooling avg) → Dropout(0.4)
      │                 → Dense(256, relu) → BatchNorm → Dropout(0.3) → 256-dim
      │
      ├──[LBP Branch]── BatchNorm → Dense(128, tanh) → BatchNorm → Dropout(0.2) → 128-dim
      │
      └──[HOG Branch]── BatchNorm → Dense(128, tanh) → BatchNorm → Dropout(0.2) → 128-dim
                   │
          [Concatenate: 256+128+128 = 512-dim]
                   │
          [Cross-Attention: Dense(512, sigmoid) × fused]
                   │
          [Fusion Proj: Dense(256, relu) → BN → Dropout(0.4)]
                   │
          [ANFIS Projection: Dense(64, tanh) → LayerNorm]
                   │
       ┌───────────────────────────────┐
       │          ANFIS CORE           │
       │ L1: Dual Fuzzification        │ ← Gaussian MF (rest dims) + Generalized Bell MF (HOG dims)
       │     5 MF per dimensi          │   center_range=2.5, sigma_init=0.5
       │ L2: Fuzzy Rule Layer          │ ← 48 rules, T-norm (softmax-weighted product)
       │ L3: Normalization Layer       │ ← w̄_k = w_k / Σw_j
       │ L4: TSK Consequent Layer      │ ← einsum, L2 regularizer 1e-5
       │ L5: LayerNorm + GELU          │ ← defuzzification + Dropout(0.3)
       └───────────────────────────────┘
                   │
       [Skip Connection: Dense(128) + Add + LayerNorm + GELU]
                   │
       ┌───────────┴──────────────────┐
       │  Emotion Head                │  Stress Head
       │  Dense(64,relu) → Drop(0.3)  │  Dense(64,relu) → Drop(0.3)
       │  Dense(32,relu)              │  Dense(32,relu)
       │  Dense(7, softmax)           │  Dense(1, sigmoid) × 100
       └──────────────────────────────┘
```

**Konfigurasi ANFIS:**

| Parameter | Nilai |
|-----------|-------|
| Fuzzy MF | 5 (Gaussian + Generalized Bell) |
| Fuzzy Rules | 48 |
| ANFIS Dim | 128 |
| Compress Dim | 64 |
| Skip Connection | ✅ |
| Layer Normalization | ✅ (3 titik) |

---

## Preprocessing Pipeline

Urutan ini identik di semua mode input (upload, capture, realtime):

```
1. Konversi ke grayscale (mendukung RGB, RGBA, grayscale)
2. Deteksi wajah — OpenCV DNN ResNet SSD (confidence > 0.5)
   → Square crop pada wajah terbesar yang ditemukan
   → Fallback: gambar utuh jika tidak ada wajah terdeteksi
3. Resize → 48×48 px
4. CLAHE (clipLimit=2.0, tileGridSize=4×4)
5. Normalize → /255.0  (float32, range [0,1])
```

**Ekstraksi fitur:**

```
LBP  — multi-radius uniform:
         R=3: P=24, 128 bin (range 0–26)
         R=5: P=16, 128 bin (range 0–18)
         Concat → 256 bin, L1-normalize

HOG  — orientations=12, pixels_per_cell=6×6, cells_per_block=2×2
         → 324-dim, L2-normalize
```

---

## Real-Time Webcam — Detail Teknis

Mode ini menggunakan `streamlit-webrtc` dengan kelas `RealtimeEmotionProcessor`.

**Optimisasi yang diterapkan:**

| Optimisasi | Detail |
|-----------|--------|
| Frame skipping | Inferensi hanya dijalankan setiap **5 frame** |
| Resolusi rendah | Frame di-resize ke **640×480** sebelum diproses |
| Face detector global | `FACE_NET` diinisialisasi sekali di level modul, tidak per-frame |
| Tanpa grafik berat | Matplotlib, LBP chart, HOG, stress gauge **tidak** dirender |
| FPS terbatas | `frameRate: {ideal: 15, max: 15}` via `media_stream_constraints` |
| Safe error handling | Semua exception ditangkap; stream tidak pernah crash |

**Smoothing prediksi:**

```
Sliding window: 8 frame terakhir → mean_probs, mean_stress
Exponential smoothing: α = 0.60
  smoothed = 0.60 × mean_current + 0.40 × smoothed_prev
```

**Overlay pada video frame:**

```
Banner atas  : "Emotion: Happy (87.3%)"
Banner bawah : "Stress: 18/100 - LOW"
Bounding box : 🟢 Hijau (Low) | 🟠 Oranye (Moderate) | 🔴 Merah (High)
```

---

## Stress Weights (Emotion → Stress)

| Emosi | Weight | Level |
|-------|--------|-------|
| Fear | 0.90 | 🔴 High |
| Angry | 0.85 | 🔴 High |
| Sad | 0.75 | 🔴 High |
| Disgust | 0.70 | 🟠 Moderate |
| Surprise | 0.45 | 🟠 Moderate |
| Neutral | 0.20 | 🟢 Low |
| Happy | 0.05 | 🟢 Low |

Stress score final = `weight × 100 + noise(±5)`, di-clip ke rentang 0–100.
Kategori: Low ≤ 33 | Moderate 34–66 | High > 66.

---

## Struktur File

```
project/
├── app.py                                  ← aplikasi utama Streamlit
├── anfis_emotion_model.weights.h5          ← weights model (opsional)
├── deploy.prototxt                         ← konfigurasi DNN wajah (WAJIB)
├── res10_300x300_ssd_iter_140000.caffemodel← bobot DNN wajah (WAJIB)
├── requirements.txt
└── README.md
```

---

## Dependencies Utama

```
streamlit
streamlit-webrtc
opencv-python
numpy
scikit-image
matplotlib
tensorflow / keras
av
Pillow
pandas
```

---

## Disclaimer

> Sistem ini merupakan **prototipe akademik** dalam mata kuliah Soft Computing.
> Output berupa *estimasi indikasi tekanan emosional* yang diturunkan dari pemetaan emosi ke
> stress weights berdasarkan pengetahuan psikologis umum, **bukan data klinis tervalidasi**.
> Sistem tidak dimaksudkan sebagai pengganti penilaian klinis oleh profesional kesehatan
> atau psikologi. Jangan gunakan hasil ini untuk diagnosis medis atau psikologis.
