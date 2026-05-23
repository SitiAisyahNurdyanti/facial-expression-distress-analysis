"""
ANFIS Hybrid Facial Emotion & Stress Analyzer
Streamlit Demo App
Tim: Isya · Ica · Awa | Soft Computing Project
"""

import os
import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import local_binary_pattern, hog
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import io
import threading
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL CACHED FACE DETECTOR (Optimization #3)
# Initialized once at module load — NOT inside preprocessing loop
# ─────────────────────────────────────────────
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

# Validasi keamanan file agar tidak memicu error crash
if os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
    FACE_NET = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
else:
    st.error("⚠️ File konfigurasi wajah (deploy.prototxt / caffemodel) belum ada di folder project!")
    st.stop()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ANFIS Hybrid Emotion & Stress Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main gradient header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99,179,237,0.2);
    }
    .main-header h1 {
        color: #63b3ed;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .main-header p {
        color: #a0aec0;
        margin: 0;
        font-size: 0.95rem;
    }
    .badge {
        display: inline-block;
        background: rgba(99,179,237,0.15);
        color: #63b3ed;
        border: 1px solid rgba(99,179,237,0.3);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-top: 0.5rem;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: linear-gradient(135deg, #2d1b1b, #3d2020);
        border: 1px solid #e53e3e;
        border-left: 4px solid #e53e3e;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box h4 {
        color: #fc8181;
        margin: 0 0 0.4rem 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box p {
        color: #fed7d7;
        margin: 0;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Emotion result card */
    .emotion-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99,179,237,0.3);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .emotion-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .confidence-score {
        font-size: 1rem;
        color: #a0aec0;
    }

    /* Stress gauge card */
    .stress-card {
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stress-low    { background: linear-gradient(135deg, #1a2e1a, #1e3a1e); border: 1px solid #48bb78; }
    .stress-mod    { background: linear-gradient(135deg, #2e2a1a, #3a341e); border: 1px solid #ed8936; }
    .stress-high   { background: linear-gradient(135deg, #2e1a1a, #3a1e1e); border: 1px solid #e53e3e; }

    /* Feature map caption */
    .feat-caption {
        background: rgba(99,179,237,0.08);
        border-left: 3px solid #63b3ed;
        padding: 0.5rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #90cdf4;
        margin-top: 0.4rem;
    }

    /* Section header */
    .section-header {
        color: #63b3ed;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(99,179,237,0.2);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0d1117;
    }
    .sidebar-section {
        background: rgba(99,179,237,0.05);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-section h4 {
        color: #63b3ed;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }
    .sidebar-section p, .sidebar-section li {
        color: #a0aec0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJIS = ['😠', '🤢', '😨', '😄', '😐', '😢', '😲']

STRESS_WEIGHTS = {
    'Angry': 0.85, 'Disgust': 0.70, 'Fear': 0.90,
    'Happy': 0.05, 'Neutral': 0.20, 'Sad': 0.75, 'Surprise': 0.45,
}
EMOTION_COLORS = {
    'Angry':    '#e53e3e',
    'Disgust':  '#805ad5',
    'Fear':     '#3182ce',
    'Happy':    '#38a169',
    'Neutral':  '#718096',
    'Sad':      '#4299e1',
    'Surprise': '#ed8936',
}
IMG_SIZE = 48
LBP_FEATURES = 256
HOG_FEATURES = 324

# ─────────────────────────────────────────────
# MODEL LOADING — lazy, cached
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the ANFIS Hybrid model weights."""
    import json
    import tensorflow as tf
    from tensorflow.keras import layers, Model, backend as K
    from tensorflow.keras.applications import MobileNetV2

    # ── Custom ANFIS Layers ──────────────────────────────────────────
    class FuzzificationLayer(layers.Layer):
        def __init__(self, n_mf=3, sigma_init=0.5, center_range=3.0, **kwargs):
            super().__init__(**kwargs)
            self.n_mf = n_mf
            self.sigma_init = sigma_init
            self.center_range = center_range

        def build(self, input_shape):
            input_dim = int(input_shape[-1])
            init_c = np.linspace(-self.center_range, self.center_range, self.n_mf).astype(np.float32)
            centers_init = np.tile(init_c, (input_dim, 1))
            self.centers = self.add_weight(
                name='centers', shape=(input_dim, self.n_mf),
                initializer=tf.constant_initializer(centers_init), trainable=True)
            self.sigmas = self.add_weight(
                name='sigmas', shape=(input_dim, self.n_mf),
                initializer=tf.constant_initializer(self.sigma_init),
                constraint=tf.keras.constraints.NonNeg(), trainable=True)
            super().build(input_shape)

        def call(self, x):
            x_exp = tf.expand_dims(x, axis=-1)
            s = tf.abs(self.sigmas) + 1e-7
            mu = tf.exp(-0.5 * tf.square((x_exp - self.centers) / s))
            return mu

        def get_config(self):
            c = super().get_config()
            c.update({'n_mf': self.n_mf, 'sigma_init': self.sigma_init, 'center_range': self.center_range})
            return c

    class GeneralizedBellMFLayer(layers.Layer):
        def __init__(self, n_mf=5, **kwargs):
            super().__init__(**kwargs)
            self.n_mf = n_mf

        def build(self, input_shape):
            input_dim = int(input_shape[-1])
            init_c = np.linspace(-2, 2, self.n_mf).astype(np.float32)
            self.centers = self.add_weight(
                name='centers', shape=(input_dim, self.n_mf),
                initializer=tf.constant_initializer(np.tile(init_c, (input_dim, 1))), trainable=True)
            self.a = self.add_weight(
                name='a', shape=(input_dim, self.n_mf),
                initializer=tf.constant_initializer(1.0),
                constraint=tf.keras.constraints.NonNeg(), trainable=True)
            self.b = self.add_weight(
                name='b', shape=(input_dim, self.n_mf),
                initializer=tf.constant_initializer(2.0),
                constraint=tf.keras.constraints.NonNeg(), trainable=True)
            super().build(input_shape)

        def call(self, x):
            x_exp = tf.expand_dims(x, axis=-1)
            a = tf.abs(self.a) + 1e-7
            b = tf.abs(self.b) + 0.5
            ratio = tf.abs((x_exp - self.centers) / a)
            mu = 1.0 / (1.0 + tf.pow(ratio, 2.0 * b))
            return mu

        def get_config(self):
            c = super().get_config()
            c['n_mf'] = self.n_mf
            return c

    class FuzzyRuleLayer(layers.Layer):
        def __init__(self, n_rules=16, n_mf=3, **kwargs):
            super().__init__(**kwargs)
            self.n_rules = n_rules
            self.n_mf = n_mf

        def build(self, input_shape):
            input_dim = int(input_shape[1])
            self.rule_weights = self.add_weight(
                name='rule_weights', shape=(self.n_rules, input_dim, self.n_mf),
                initializer='glorot_uniform', trainable=True)
            super().build(input_shape)

        def call(self, mu, training=None):
            rw_soft = tf.nn.softmax(self.rule_weights, axis=-1)
            mu_exp = tf.expand_dims(mu, axis=1)
            selected = tf.reduce_sum(mu_exp * rw_soft, axis=-1)
            w = tf.reduce_prod(selected + 1e-7, axis=-1)
            return w

        def get_config(self):
            c = super().get_config()
            c.update({'n_rules': self.n_rules, 'n_mf': self.n_mf})
            return c

    class NormalizationLayer(layers.Layer):
        def call(self, w):
            w_sum = tf.reduce_sum(w, axis=-1, keepdims=True) + 1e-7
            return w / w_sum

    class ConsequentLayer(layers.Layer):
        def __init__(self, n_rules, input_dim, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.n_rules = n_rules
            self.input_dim = input_dim
            self.output_dim = output_dim

        def build(self, input_shape):
            self.p = self.add_weight(
                name='consequent_p',
                shape=(self.n_rules, self.input_dim + 1, self.output_dim),
                initializer='glorot_uniform',
                regularizer=tf.keras.regularizers.l2(1e-5),
                trainable=True)
            super().build(input_shape)

        def call(self, inputs):
            w_bar, x = inputs
            bias = tf.ones_like(x[:, :1])
            x_aug = tf.concat([x, bias], axis=1)
            fk = tf.einsum('bd,kdo->bko', x_aug, self.p)
            w_exp = tf.expand_dims(w_bar, axis=-1)
            out = tf.reduce_sum(w_exp * fk, axis=1)
            return out

        def get_config(self):
            c = super().get_config()
            c.update({'n_rules': self.n_rules, 'input_dim': self.input_dim, 'output_dim': self.output_dim})
            return c

    # ── Config ───────────────────────────────────────────────────────
    class Cfg:
        IMG_SIZE        = 48
        CHANNELS        = 1
        NUM_CLASSES     = 7
        LBP_FEATURES    = 256
        HOG_FEATURES    = 324
        CNN_FEATURES    = 256
        ANFIS_RULES     = 48
        FUZZY_MF        = 5
        ANFIS_DIM       = 128
        COMPRESS_DIM    = 64
        CNN_FINETUNE_LAYERS = 30

    cfg = Cfg()

    # ── Build model ───────────────────────────────────────────────────
    def build_cnn_branch(input_tensor):
        x = layers.Resizing(96, 96)(input_tensor)
        x = layers.Concatenate()([x, x, x])
        base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=None, pooling='avg')
        base_model.trainable = False
        x = base_model(x, training=False)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(cfg.CNN_FEATURES, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        return x

    def build_lbp_branch(input_tensor):
        x = layers.BatchNormalization(name='lbp_bn')(input_tensor)
        x = layers.Dense(128, activation='tanh', name='lbp_project')(x)
        x = layers.BatchNormalization(name='lbp_project_bn')(x)
        x = layers.Dropout(0.2, name='lbp_dropout')(x)
        return x

    def build_hog_branch(input_tensor):
        x = layers.BatchNormalization(name='hog_bn')(input_tensor)
        x = layers.Dense(128, activation='tanh', name='hog_compress')(x)
        x = layers.BatchNormalization(name='hog_compress_bn')(x)
        x = layers.Dropout(0.2, name='hog_dropout')(x)
        return x

    img_input = layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS), name='input_image')
    lbp_input = layers.Input(shape=(cfg.LBP_FEATURES,), name='input_lbp')
    hog_input = layers.Input(shape=(cfg.HOG_FEATURES,), name='input_hog')

    cnn_feat = build_cnn_branch(img_input)
    lbp_feat = build_lbp_branch(lbp_input)
    hog_feat = build_hog_branch(hog_input)

    fused = layers.Concatenate(name='feature_fusion')([cnn_feat, lbp_feat, hog_feat])
    fused_dim = 256 + 128 + 128
    attn = layers.Dense(fused_dim, activation='sigmoid', name='cross_attention')(fused)
    fused = layers.Multiply(name='attended_features')([fused, attn])

    z = layers.Dense(256, activation='relu', name='fusion_proj_1')(fused)
    z = layers.BatchNormalization(name='fusion_bn_1')(z)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(cfg.COMPRESS_DIM, activation='tanh', name='anfis_projection')(z)
    z = layers.LayerNormalization(name='anfis_layernorm')(z)

    hog_dim  = max(1, cfg.COMPRESS_DIM // 3)
    rest_dim = cfg.COMPRESS_DIM - hog_dim
    z_hog  = layers.Lambda(lambda t: t[:, :hog_dim],  name='split_hog_dims')(z)
    z_rest = layers.Lambda(lambda t: t[:, hog_dim:], name='split_rest_dims')(z)

    mu_hog  = GeneralizedBellMFLayer(n_mf=cfg.FUZZY_MF, name='l1_fuzz_hog_bell')(z_hog)
    mu_rest = FuzzificationLayer(n_mf=cfg.FUZZY_MF, sigma_init=0.5, center_range=2.5, name='l1_fuzz_standard')(z_rest)

    mu_hog_flat  = layers.Reshape((hog_dim * cfg.FUZZY_MF,),  name='reshape_mu_hog')(mu_hog)
    mu_rest_flat = layers.Reshape((rest_dim * cfg.FUZZY_MF,), name='reshape_mu_rest')(mu_rest)
    mu_combined  = layers.Concatenate(name='mu_concat')([mu_hog_flat, mu_rest_flat])

    unified_mf_dim = cfg.COMPRESS_DIM * cfg.FUZZY_MF
    mu_unified  = layers.Dense(unified_mf_dim, activation='sigmoid', name='mu_unification')(mu_combined)
    mu_reshaped = layers.Reshape((cfg.COMPRESS_DIM, cfg.FUZZY_MF), name='mu_reshape')(mu_unified)

    w     = FuzzyRuleLayer(n_rules=cfg.ANFIS_RULES, n_mf=cfg.FUZZY_MF, name='l2_fuzzy_rules')(mu_reshaped)
    w_bar = NormalizationLayer(name='l3_normalization')(w)
    anfis_out = ConsequentLayer(
        n_rules=cfg.ANFIS_RULES, input_dim=cfg.COMPRESS_DIM, output_dim=cfg.ANFIS_DIM,
        name='l4_consequent')([w_bar, z])
    anfis_out = layers.LayerNormalization(name='l5_defuzz_layernorm')(anfis_out)
    anfis_out = layers.Activation('gelu', name='l5_defuzzification')(anfis_out)
    anfis_out = layers.Dropout(0.3)(anfis_out)

    fused_proj = layers.Dense(cfg.ANFIS_DIM, use_bias=False, name='skip_projection')(fused)
    combined   = layers.Add(name='residual_connection')([anfis_out, fused_proj])
    combined   = layers.LayerNormalization(name='combined_layernorm')(combined)
    combined   = layers.Activation('gelu', name='combined_gelu')(combined)

    emo = layers.Dense(64, activation='relu', name='emotion_dense1')(combined)
    emo = layers.Dropout(0.3)(emo)
    emo = layers.Dense(32, activation='relu', name='emotion_dense2')(emo)
    emotion_out = layers.Dense(cfg.NUM_CLASSES, activation='softmax', name='emotion_output')(emo)

    stress = layers.Dense(64, activation='relu', name='stress_dense1')(combined)
    stress = layers.Dropout(0.3)(stress)
    stress = layers.Dense(32, activation='relu', name='stress_dense2')(stress)
    stress_out = layers.Dense(1, activation='sigmoid', name='stress_output')(stress)

    model = Model(
        inputs=[img_input, lbp_input, hog_input],
        outputs=[emotion_out, stress_out],
        name='ANFIS_Hybrid_FacialEmotion')

    # Try loading weights if available
    weights_path = 'anfis_emotion_model.weights.h5'
    loaded = False
    if os.path.exists(weights_path):
        try:
            print ("Modelnya ada nih")
            model.load_weights(weights_path)
            loaded = True
        except Exception:
            loaded = False

    return model, loaded

# ─────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────
def apply_clahe(img_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(img_uint8)

def preprocess_image(img_array):
    """Full preprocessing pipeline with automatic Face Cropping."""

    # Ambil dimensi asli gambar (Tinggi dan Lebar) untuk kalkulasi koordinat piksel
    h, w = img_array.shape[0], img_array.shape[1]

    # 1. Menyiapkan citra berwarna untuk DNN dan grayscale untuk CLAHE/Ekstraksi Fitur
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        img_rgb = img_array.copy()
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        gray = img_array.copy()
        
    gray = gray.astype(np.uint8)

    # 2. USE GLOBAL CACHED FACE DETECTOR (Optimization #3 — Neural Network DNN)
    try:
        # Konversi citra RGB ke format BLOB BGR yang dipahami oleh OpenCV DNN
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)), 
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        FACE_NET.setInput(blob)
        detections = FACE_NET.forward()

        face_found = False
        largest_area = 0
        best_box = None

        # Iterasi hasil deteksi wajah untuk mencari wajah terbesar
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Saring wajah dengan ambang keyakinan di atas 50%
            if confidence > 0.5:
                # Mengubah koordinat relatif desimal (0.0 - 1.0) menjadi piksel asli
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x_start, y_start, x_end, y_end) = box.astype("int")

                # Batasi koordinat agar tetap di dalam dimensi frame citra asli
                x_start, y_start = max(0, x_start), max(0, y_start)
                x_end, y_end = min(w, x_end), min(h, y_end)

                face_w = x_end - x_start
                face_h = y_end - y_start
                area = face_w * face_h

                # Menyimpan wajah utama (luas terbesar)
                if area > largest_area:
                    largest_area = area
                    best_box = (x_start, y_start, face_w, face_h)
                    face_found = True

        # 3. CROP TO FACE IF FOUND
        if face_found and best_box is not None:
            # Ambil koordinat wajah hasil seleksi DNN
            x, y, face_w, face_h = best_box
            
            # Make it a perfect square bounding box to avoid stretching distortion
            side = max(face_w, face_h)
            cx, cy = x + face_w // 2, y + face_h // 2
            
            # Calculate new bounds while keeping it within image boundaries
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
            
            # Crop the image to just the face bounding box dari citra grayscale
            gray = gray[y1:y2, x1:x2]

    except Exception as e:
        # Fallback safe: jika pemrosesan DNN gagal, gunakan citra utuh agar tidak crash
        pass

    # 4. Resume normal pipeline on the cropped face area
    if gray.size > 0:
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        enhanced = apply_clahe(resized)
        normalized = enhanced.astype(np.float32) / 255.0
        return normalized
    else:
        # Penanganan darurat jika terjadi crop kosong
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

def extract_lbp_features(img_2d, n_bins=256):
    img_uint8 = (img_2d * 255).astype(np.uint8)
    lbp3 = local_binary_pattern(img_uint8, 24, 3, method='uniform')
    hist3, _ = np.histogram(lbp3.ravel(), bins=128, range=(0, 26))
    lbp5 = local_binary_pattern(img_uint8, 16, 5, method='uniform')
    hist5, _ = np.histogram(lbp5.ravel(), bins=128, range=(0, 18))
    hist = np.concatenate([hist3, hist5]).astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    if len(hist) < n_bins:
        hist = np.pad(hist, (0, n_bins - len(hist)))
    else:
        hist = hist[:n_bins]
    return hist

def extract_lbp_map(img_2d):
    """Return LBP visual map for display."""
    img_uint8 = (img_2d * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, 24, 3, method='uniform')
    return lbp

def extract_hog_features(img_2d, target_size=324):
    img_uint8 = (img_2d * 255).astype(np.uint8)
    feat, hog_img = hog(
        img_uint8, orientations=12,
        pixels_per_cell=(6, 6), cells_per_block=(2, 2),
        visualize=True, feature_vector=True)
    if len(feat) < target_size:
        feat = np.pad(feat, (0, target_size - len(feat)))
    else:
        feat = feat[:target_size]
    feat = feat.astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-7
    return feat / norm, hog_img

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────
# PREDICTION (with fallback if no weights)
# ─────────────────────────────────────────────
def predict(model, img_normalized, lbp_feat, hog_feat, model_loaded):
    if model_loaded:
        img_cnn = img_normalized[np.newaxis, ..., np.newaxis]
        lbp_in  = lbp_feat[np.newaxis, :]
        hog_in  = hog_feat[np.newaxis, :]
        try:
            emo_probs, stress_val = model([img_cnn, lbp_in, hog_in], training=False)
            probs = emo_probs.numpy()[0]
            stress = float(stress_val.numpy()[0][0]) * 100
            return probs, stress
        except Exception as e:
            st.warning(f"Prediksi model gagal: {e}. Menggunakan simulasi.")
    # Simulasi berbasis LBP/HOG jika model belum di-load
    rng = np.random.default_rng(int(np.sum(lbp_feat[:10] * 1000)) % 2**31)
    probs = rng.dirichlet(np.ones(7) * 2)
    dominant = np.argmax(probs)
    label = EMOTION_LABELS[dominant]
    stress_base = STRESS_WEIGHTS[label]
    stress = float(np.clip(stress_base * 100 + rng.normal(0, 5), 0, 100))
    return probs, stress

def stress_level_label(score):
    if score <= 33:
        return "Low", "🟢", "stress-low"
    elif score <= 66:
        return "Moderate", "🟠", "stress-mod"
    else:
        return "High", "🔴", "stress-high"

# ─────────────────────────────────────────────
# REAL-TIME WEBCAM PROCESSOR  (NEW FEATURE)
# Uses streamlit-webrtc + existing pipeline
# All optimizations applied:
#   #1 Frame skipping (every 5th frame)
#   #2 Lower resolution (640×480)
#   #3 Global FACE_CASCADE (already patched above)
#   #4 No matplotlib / heavy charts
#   #5 FPS limited via media constraints (passed to webrtc_streamer)
#   #6 Safe error handling — never crashes stream
# ─────────────────────────────────────────────
class RealtimeEmotionProcessor:
    """
    VideoProcessorBase-compatible class for streamlit-webrtc.
    Receives raw av.VideoFrame, runs ANFIS inference every 5th frame,
    overlays results, and returns the annotated frame.
    """

    def __init__(self, model=None, mdl_loaded=False):   
        self.model = model
        self.mdl_loaded = mdl_loaded

        # Optimisation #1 — frame counter for skipping
        self.frame_count: int = 0

        # Cached last prediction (reused on skipped frames)
        self._lock = threading.Lock()
        self._last_label: str     = "Detecting..."
        self._last_conf: float    = 0.0
        self._last_stress: float  = 0.0
        self._last_stress_lv: str = ""
        self._last_faces           = []   # list of (x,y,w,h)

        self.probs_history = []
        self.stress_history = []
        self.history_window = 8

        # Memory tracking arrays initialized to balanced defaults
        self._smoothed_probs = np.ones(7) / 7
        self._smoothed_stress = 0.0
        self.ALPHA = 0.60

    # ------------------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Called for every incoming webcam frame by streamlit-webrtc."""
        # Convert to numpy BGR (standard OpenCV format)
        img_bgr = frame.to_ndarray(format="bgr24")

        # Optimisation #2 — downscale BEFORE any processing
        img_bgr = cv2.resize(img_bgr, (640, 480))

        self.frame_count += 1

        # Optimisation #1 — only run heavy inference every 5th frame
        if self.frame_count % 5 == 0:
            self._run_inference(img_bgr)

        # Draw overlay on every frame (uses cached results)
        annotated = self._draw_overlay(img_bgr)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # ------------------------------------------------------------------
    def _run_inference(self, img_bgr: np.ndarray):
        """
        Run face detection (OpenCV DNN) + ANFIS prediction.
        Updates internal cached result state. Safe — never raises.
        """
        try:
            h, w, _ = img_bgr.shape

            # 1. Preprocessing input khusus untuk wajah berbasis Neural Network (DNN)
            # Mengubah ukuran ke 300x300 dan menyelaraskan rata-rata warna (Mean Subtraction)
            blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            FACE_NET.setInput(blob)
            detections = FACE_NET.forward()

            face_found = False
            largest_area = 0
            best_box = None

            # 2. Iterasi hasil pencarian wajah yang ditemukan
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Ambil deteksi yang tingkat keyakinannya di atas 50%
                if confidence > 0.5:
                    # Ambil koordinat pembatas relatif (0.0 - 1.0)
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x_start, y_start, x_end, y_end) = box.astype("int")

                    # Batasi koordinat agar berada di dalam dimensi frame kamera
                    x_start, y_start = max(0, x_start), max(0, y_start)
                    x_end, y_end = min(w, x_end), min(h, y_end)

                    face_w = x_end - x_start
                    face_h = y_end - y_start
                    area = face_w * face_h

                    # Filter wajah utama (terbesar) yang paling dekat kamera
                    if area > largest_area:
                        largest_area = area
                        best_box = (x_start, y_start, face_w, face_h)
                        face_found = True

            if not face_found or best_box is None:
                with self._lock:
                    self._last_label    = "No face detected"
                    self._last_conf     = 0.0
                    self._last_faces    = []
                return

            x, y, face_w, face_h = best_box

            # 3. Square Bounding Box Logic agar tidak peyang saat diekstrak
            side = max(face_w, face_h)
            cx, cy = x + face_w // 2, y + face_h // 2

            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

            face_region_bgr = img_bgr[y1:y2, x1:x2]

            if face_region_bgr.size == 0:
                return

            face_region_rgb = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)
            # Resizing ke 48x48 untuk Tensor ANFIS kalian
            resized = cv2.resize(face_region_rgb, (IMG_SIZE, IMG_SIZE))
            gray_face = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            enhanced = apply_clahe(gray_face)
            img_norm = enhanced.astype(np.float32) / 255.0

            # Feature Extractors (LBP & HOG)
            lbp_feat = extract_lbp_features(img_norm)
            hog_feat, _ = extract_hog_features(img_norm)

            # Jalankan model prediksi ANFIS
            raw_probs, raw_stress = predict(self.model, img_norm, lbp_feat, hog_feat, self.mdl_loaded)

            self.probs_history.append(raw_probs)
            if len(self.probs_history) > self.history_window:
                self.probs_history.pop(0)
            mean_probs = np.mean(self.probs_history, axis=0)

            self.stress_history.append(raw_stress)
            if len(self.stress_history) > self.history_window:
                self.stress_history.pop(0)
            mean_stress = np.mean(self.stress_history)

            # Exponential Smoothing
            self._smoothed_probs = (self.ALPHA * mean_probs) + ((1 - self.ALPHA) * self._smoothed_probs)
            self._smoothed_stress = (self.ALPHA * mean_stress) + ((1 - self.ALPHA) * self._smoothed_stress)

            top_idx   = int(np.argmax(self._smoothed_probs))
            top_label = EMOTION_LABELS[top_idx]
            top_conf  = float(self._smoothed_probs[top_idx])
            stress_lv, _, _ = stress_level_label(self._smoothed_stress)

            with self._lock:
                self._last_label    = top_label
                self._last_conf     = top_conf
                self._last_stress   = self._smoothed_stress
                self._last_stress_lv = stress_lv
                self._last_faces    = [(x, y, face_w, face_h)]

        except Exception as e:
            with self._lock:
                self._last_label    = "Error — retrying"
                self._last_conf     = 0.0
                self._last_stress   = 0.0
                self._last_stress_lv = ""
                self._last_faces    = []

    # ------------------------------------------------------------------
    def _draw_overlay(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Draw bounding box + prediction overlay on the frame.
        Colour-codes bounding box by stress level:
          🟢 Green  → Low   (stress ≤ 33)
          🟠 Orange → Moderate (33 < stress ≤ 66)
          🔴 Red    → High  (stress > 66)
        """
        with self._lock:
            label    = self._last_label
            conf     = self._last_conf
            stress   = self._last_stress
            stress_lv = self._last_stress_lv
            faces    = list(self._last_faces)

        out = img_bgr.copy()

        # ── Bounding box ───────────────────────────────────────────
        if faces:
            x, y, w, h = faces[0]
            if stress_lv == "Low":
                box_color = (72, 187, 120)    # Green  (BGR)
            elif stress_lv == "Moderate":
                box_color = (54, 137, 237)    # Orange (BGR)
            else:
                box_color = (62, 62, 229)     # Red    (BGR)
            cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)

        # ── Semi-transparent label banner at top ──────────────────
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (640, 64), (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # ── Text lines ────────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_small = cv2.FONT_HERSHEY_SIMPLEX

        if label in ("No face detected", "Detecting...", "Error — retrying"):
            # Fallback message
            cv2.putText(out, label, (12, 40), font, 0.75,
                        (160, 160, 160), 1, cv2.LINE_AA)
        else:
            # Emotion + confidence
            emo_text = f"Emotion: {label} ({conf*100:.1f}%)"
            cv2.putText(out, emo_text, (10, 26), font, 0.70,
                        (227, 232, 240), 1, cv2.LINE_AA)

            # Stress score + level
            stress_text = f"Stress: {stress:.0f}/100 - {stress_lv.upper()}"
            stress_color = (72, 187, 120) if stress_lv == "Low" else \
                           (54, 137, 237) if stress_lv == "Moderate" else \
                           (62, 62, 229)
            cv2.putText(out, stress_text, (10, 54), font_small, 0.60,
                        stress_color, 1, cv2.LINE_AA)

        return out

# ─────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────
def plot_probability_bar(probs):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    colors = [EMOTION_COLORS[e] for e in EMOTION_LABELS]
    y_pos  = np.arange(len(EMOTION_LABELS))
    labels_with_emoji = [f"{EMOTION_EMOJIS[i]}  {EMOTION_LABELS[i]}" for i in range(7)]

    bars = ax.barh(y_pos, probs * 100, color=colors, alpha=0.85,
                   height=0.6, edgecolor='none')

    # Highlight max
    max_idx = np.argmax(probs)
    bars[max_idx].set_alpha(1.0)
    bars[max_idx].set_edgecolor('white')
    bars[max_idx].set_linewidth(1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_with_emoji, color='#e2e8f0', fontsize=10)
    ax.set_xlabel('Probability (%)', color='#a0aec0', fontsize=9)
    ax.set_xlim(0, 100)
    ax.tick_params(axis='x', colors='#718096', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#2d3748')
    ax.spines['left'].set_color('#2d3748')

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob * 100 + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob*100:.1f}%', va='center', ha='left',
                color='#e2e8f0', fontsize=8.5)

    ax.set_title('Distribusi Probabilitas 7 Emosi', color='#63b3ed',
                 fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout(pad=0.8)
    return fig

def plot_stress_gauge(stress_score):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={'aspect': 'equal'})
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Semicircle background segments
    import matplotlib.patches as mpatches
    from matplotlib.patches import Wedge

    segments = [
        (0, 60,  '#48bb7844'),  # Low   (0–33%)
        (60, 120, '#ed893644'), # Mod
        (120, 180, '#e53e3e44'),# High
    ]
    for start, end, color in segments:
        wedge = Wedge((0.5, 0.1), 0.42, 180 - end, 180 - start,
                      width=0.12, facecolor=color, edgecolor='none', transform=ax.transAxes)
        ax.add_patch(wedge)

    # Needle
    angle_rad = np.pi * (1 - stress_score / 100)
    needle_len = 0.35
    cx, cy = 0.5, 0.1
    nx = cx + needle_len * np.cos(angle_rad)
    ny = cy + needle_len * np.sin(angle_rad)
    ax.annotate('', xy=(nx, ny), xytext=(cx, cy),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

    # Pivot dot
    ax.plot(cx, cy, 'o', color='white', markersize=8, transform=ax.transAxes,
            zorder=5)

    # Score text
    ax.text(0.5, 0.55, f'{stress_score:.0f}', transform=ax.transAxes,
            ha='center', va='center', fontsize=28, fontweight='bold',
            color='white')
    ax.text(0.5, 0.38, '/ 100', transform=ax.transAxes,
            ha='center', va='center', fontsize=11, color='#a0aec0')

    # Level labels
    for x, label, color in [(0.05, 'LOW', '#48bb78'), (0.5, 'MOD', '#ed8936'), (0.95, 'HIGH', '#e53e3e')]:
        ax.text(x, -0.05, label, transform=ax.transAxes,
                ha='center', va='center', fontsize=7, color=color, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')
    ax.set_title('Stress Level Gauge', color='#63b3ed',
                 fontsize=10, fontweight='bold', pad=4)
    plt.tight_layout(pad=0.5)
    return fig

def plot_lbp_map(lbp_img, img_orig):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.axis('off')

    axes[0].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original (48×48)', color='#a0aec0', fontsize=9)

    axes[1].imshow(lbp_img, cmap='hot')
    axes[1].set_title('LBP Map\n(Tekstur Kerutan)', color='#a0aec0', fontsize=9)

    plt.tight_layout(pad=0.5)
    return fig

def plot_hog_map(hog_img, img_orig):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        ax.axis('off')

    axes[0].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original (48×48)', color='#a0aec0', fontsize=9)

    axes[1].imshow(hog_img, cmap='inferno')
    axes[1].set_title('HOG Map\n(Gradien Kontur Wajah)', color='#a0aec0', fontsize=9)

    plt.tight_layout(pad=0.5)
    return fig

def plot_lbp_histogram(lbp_feat):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.bar(np.arange(64), lbp_feat[:64], color='#63b3ed', alpha=0.8, width=1.0, edgecolor='none')
    ax.set_xlabel('Bin', color='#a0aec0', fontsize=8)
    ax.set_ylabel('Normalized Freq', color='#a0aec0', fontsize=8)
    ax.set_title('LBP Histogram (64 of 256 bins)', color='#63b3ed', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#718096', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#2d3748')
    plt.tight_layout(pad=0.5)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-section'>
    <h4>🧠 Tentang Model</h4>
    <p><b>ANFIS Hybrid</b> menggabungkan:</p>
    <ul>
        <li>CNN (MobileNetV2) → fitur spasial 256-dim</li>
        <li>LBP → tekstur kerutan 256-dim</li>
        <li>HOG → gradien kontur 324-dim</li>
        <li>ANFIS Core (48 rules, 5 MF)</li>
    </ul>
    <p>Dataset: <b>FER2013</b> — 35.887 gambar wajah 48×48 px</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-section'>
    <h4>📊 Stress Weights</h4>
    """, unsafe_allow_html=True)
    for em, w in sorted(STRESS_WEIGHTS.items(), key=lambda x: -x[1]):
        lvl = "🔴" if w >= 0.75 else ("🟠" if w >= 0.45 else "🟢")
        st.markdown(f"<p>{lvl} <b>{em}</b>: {w*100:.0f}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-section'>
    <h4>⚙️ Pipeline Preprocessing</h4>
    <p>
    1. Convert ke grayscale<br>
    2. Resize → 48×48 px<br>
    3. CLAHE (clipLimit=2.0)<br>
    4. Normalize /255.0<br>
    5. Ekstrak LBP (multi-radius)<br>
    6. Ekstrak HOG (12 orient.)<br>
    7. ANFIS Inference
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-section'>
    <h4>👥 Tim Pengembang</h4>
    <p>
    🧑‍💻 <b>Isya</b> <br>
    🧑‍💻 <b>Ica</b> <br>
    🧑‍💻 <b>Awa</b> <br><br>
    Mata Kuliah: Soft Computing<br>
    Dataset: FER2013 (Kaggle)
    </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class='main-header'>
    <h1>🧠 ANFIS Hybrid Facial Emotion & Stress Analyzer</h1>
    <p>Analisis Indikasi Tekanan Emosional Berdasarkan Ekspresi Wajah</p>
    <span class='badge'>CNN + LBP + HOG</span>
    <span class='badge'>ANFIS Core (48 Rules)</span>
    <span class='badge'>7 Emosi</span>
    <span class='badge'>FER2013</span>
    <span class='badge'>Soft Computing Project</span>
</div>
""", unsafe_allow_html=True)

# Disclaimer (wajib tampil)
st.markdown("""
<div class='disclaimer-box'>
    <h4>⚠️ Disclaimer — Baca Sebelum Menggunakan</h4>
    <p>
    Sistem ini merupakan <b>prototipe akademik</b> dalam mata kuliah <b>Soft Computing</b>.
    Output berupa <em>estimasi indikasi tekanan emosional</em> yang diturunkan dari pemetaan emosi ke
    stress weights berdasarkan pengetahuan psikologis umum, <b>bukan data klinis tervalidasi</b>.
    Sistem <u>tidak dimaksudkan sebagai pengganti penilaian klinis</u> oleh profesional kesehatan
    atau psikologi. Jangan gunakan hasil ini untuk diagnosis medis atau psikologis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load model (with spinner) ──
with st.spinner("Memuat model ANFIS Hybrid..."):
    model, model_loaded = load_model()

if model_loaded:
    st.success("✅ Model berhasil dimuat dari weights file.")
else:
    st.info(
        "ℹ️ **File weights tidak ditemukan.** Pastikan file `anfis_emotion_model.weights.h5` "
        "ada di direktori yang sama dengan `app.py`. Saat ini menggunakan **mode simulasi** "
        "berbasis LBP/HOG untuk demo preprocessing pipeline."
    )

st.markdown("---")

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>📸 Input Gambar Wajah</div>", unsafe_allow_html=True)

input_mode = st.radio(
    "Pilih sumber input:",
    ["📁 Upload Gambar (JPG/PNG)", "📷 Capture dari Webcam", "🎥 Real-Time Webcam Detection"],
    horizontal=True
)

uploaded_img = None
raw_img_array = None

if input_mode == "📁 Upload Gambar (JPG/PNG)":
    uploaded_file = st.file_uploader(
        "Upload gambar wajah", type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG. Pastikan wajah terlihat jelas."
    )
    if uploaded_file is not None:
        from PIL import Image
        pil_img = Image.open(uploaded_file)
        raw_img_array = np.array(pil_img)
        st.image(pil_img, caption="Gambar yang diupload", width=200)

elif input_mode == "📷 Capture dari Webcam":
    cam_img = st.camera_input("Ambil foto dari webcam")
    if cam_img is not None:
        from PIL import Image
        pil_img = Image.open(cam_img)
        raw_img_array = np.array(pil_img)
        st.image(pil_img, caption="Foto dari webcam", width=200)

else:
    # ── 🎥 REAL-TIME WEBCAM DETECTION (NEW FEATURE) ──────────────────
    st.markdown("""
    <div style='
        background: rgba(99,179,237,0.07);
        border: 1px solid rgba(99,179,237,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    '>
        <div style='color:#63b3ed; font-weight:600; font-size:0.95rem; margin-bottom:0.4rem;'>
            🎥 Real-Time Webcam Detection
        </div>
        <div style='color:#a0aec0; font-size:0.83rem; line-height:1.6;'>
            Mode ini menjalankan prediksi ANFIS <b>langsung pada stream webcam</b> — tanpa perlu
            mengambil foto terlebih dahulu.<br>
            ✅ Inferensi setiap 5 frame &nbsp;|&nbsp; ✅ Resolusi 640×480 &nbsp;|&nbsp;
            ✅ Bounding box warna stress &nbsp;|&nbsp; ✅ Overlay emosi + stress realtime
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cache model reference into session_state so the processor thread can access it
    st.session_state["rt_model"]        = model
    st.session_state["rt_model_loaded"] = model_loaded

    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="realtime_anfis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: RealtimeEmotionProcessor(
            model=st.session_state.get("rt_model", None),
            mdl_loaded=st.session_state.get("rt_model_loaded", False)
        ),
        # Optimisation #5 — limit webcam FPS + resolution
        media_stream_constraints={
            "video": {
                "width":     {"ideal": 640},
                "height":    {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("""
    <div style='color:#718096; font-size:0.8rem; margin-top:0.6rem;'>
        ℹ️ Klik <b>START</b> untuk memulai stream. Izinkan akses kamera saat diminta browser.
        Overlay emosi &amp; stress tampil langsung di atas video.
        Visualisasi grafik tidak ditampilkan di mode realtime untuk menjaga performa.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
if raw_img_array is not None:
    st.markdown("---")

    with st.spinner("🔄 Menjalankan preprocessing & inferensi ANFIS..."):
        # Preprocessing
        img_norm = preprocess_image(raw_img_array)
        st.image(img_norm, caption="Hasil Preprocessing Mask Wajah (Model View)", width=100)
        lbp_feat = extract_lbp_features(img_norm)
        lbp_map  = extract_lbp_map(img_norm)
        hog_feat, hog_map = extract_hog_features(img_norm)

        # Prediction
        probs, stress_score = predict(model, img_norm, lbp_feat, hog_feat, model_loaded)
        top_idx   = np.argmax(probs)
        top_label = EMOTION_LABELS[top_idx]
        top_conf  = probs[top_idx]
        stress_lv, stress_icon, stress_cls = stress_level_label(stress_score)

    # ── RESULTS SECTION ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>🎯 Hasil Analisis</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    with col1:
        emotion_color = EMOTION_COLORS[top_label]
        st.markdown(f"""
        <div class='emotion-card'>
            <div style='color:#a0aec0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;'>Prediksi Emosi</div>
            <div class='emotion-label' style='color:{emotion_color};'>{EMOTION_EMOJIS[top_idx]} {top_label}</div>
            <div class='confidence-score'>Confidence: <b style='color:white;'>{top_conf*100:.1f}%</b></div>
            <div style='margin-top:0.8rem;'>
                <div style='background:#1a1a2e; border-radius:8px; height:8px; overflow:hidden;'>
                    <div style='width:{top_conf*100:.1f}%; background:{emotion_color}; height:100%; border-radius:8px; transition:width 0.5s;'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='stress-card {stress_cls}'>
            <div style='color:#a0aec0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;'>Stress Level</div>
            <div style='font-size:2.8rem; font-weight:800; color:white; margin:0.2rem 0;'>{stress_score:.0f}</div>
            <div style='font-size:0.85rem; color:#a0aec0;'>/ 100</div>
            <div style='font-size:1.3rem; margin-top:0.4rem;'>{stress_icon} <b style='color:white;'>{stress_lv}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Top 3 emotions
        top3_idx = np.argsort(probs)[::-1][:3]
        st.markdown("<div style='padding:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='color:#63b3ed; font-size:0.9rem; font-weight:600; margin-bottom:0.5rem;'>Top 3 Prediksi</div>", unsafe_allow_html=True)
        for rank, idx in enumerate(top3_idx, 1):
            st.markdown(
                f"<div style='color:#e2e8f0; font-size:0.9rem; margin-bottom:0.3rem;'>"
                f"<b style='color:{EMOTION_COLORS[EMOTION_LABELS[idx]]};'>#{rank}</b> "
                f"{EMOTION_EMOJIS[idx]} {EMOTION_LABELS[idx]} — "
                f"<b>{probs[idx]*100:.1f}%</b></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Distribusi Probabilitas & Stress Gauge</div>",
                unsafe_allow_html=True)

    col_bar, col_gauge = st.columns([3, 1.8])

    with col_bar:
        fig_bar = plot_probability_bar(probs)
        st.pyplot(fig_bar, width="stretch")

    with col_gauge:
        fig_gauge = plot_stress_gauge(stress_score)
        st.pyplot(fig_gauge, width="stretch")

    # ── INTERMEDIATE FEATURES ────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>🔬 Intermediate Features — Transparansi Pipeline</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#a0aec0; font-size:0.85rem;'>
    Visualisasi fitur matematika kerutan wajah yang diekstrak sebelum masuk ke ANFIS Core.
    LBP menangkap <em>tekstur kulit</em>, HOG menangkap <em>orientasi gradien otot wajah</em>.
    </p>
    """, unsafe_allow_html=True)

    col_lbp, col_hog = st.columns(2)

    with col_lbp:
        st.markdown("**🔴 LBP Map — Local Binary Pattern (Tekstur Kerutan)**")
        fig_lbp = plot_lbp_map(lbp_map, img_norm)
        st.pyplot(fig_lbp, use_container_width=True)
        st.markdown("""
        <div class='feat-caption'>
        Formula: LBP(p,r) = Σ s(gn−gc)·2ⁿ | n_points=24, radius=3, uniform method.
        Multi-radius: R=3 (256 bin detail halus) + R=5 (256 bin struktur besar).
        </div>
        """, unsafe_allow_html=True)

    with col_hog:
        st.markdown("**🟠 HOG Map — Histogram of Oriented Gradients (Kontur Wajah)**")
        fig_hog = plot_hog_map(hog_map, img_norm)
        st.pyplot(fig_hog, use_container_width=True)
        st.markdown("""
        <div class='feat-caption'>
        Formula: H(θ) = Σ‖∇I(x,y)‖·δ(angle=θ) | orientations=12, ppc=6×6, cpb=2×2.
        Menangkap Action Units (AU) dan kontur otot ekspresi wajah.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_hist, col_meta = st.columns([2, 1])

    with col_hist:
        st.markdown("**📈 LBP Histogram (64 dari 256 bin)**")
        fig_hist = plot_lbp_histogram(lbp_feat)
        st.pyplot(fig_hist, use_container_width=True)

    with col_meta:
        st.markdown("**📐 Dimensi Fitur**")
        feat_data = {
            "Branch": ["CNN", "LBP", "HOG", "Fused"],
            "Dimensi": ["256", "128 (proj)", "128 (proj)", "512"],
            "Method": ["MobileNetV2", "Multi-radius", "12 orient.", "Concat+Attn"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(feat_data), hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div style='background:rgba(99,179,237,0.08); border-radius:8px; padding:0.8rem; margin-top:0.8rem;'>
            <div style='color:#63b3ed; font-size:0.85rem; font-weight:600;'>ANFIS Config</div>
            <div style='color:#a0aec0; font-size:0.8rem; margin-top:0.3rem;'>
                Rules: 48 | MF: 5 (Gaussian + Bell)<br>
                ANFIS Dim: 128 | Compress: 64<br>
                Skip Connection: ✅ | Layer Norm: ✅
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ARCHITECTURE EXPLAINER ───────────────────────────────────────
    with st.expander("📐 Arsitektur Pipeline (Detail)", expanded=False):
        st.markdown("""
        ```
        Input Image (JPG/PNG)
              │
              ├──[Preprocessing]── grayscale → resize(48×48) → CLAHE → /255
              │
              ├──[CNN Branch]── MobileNetV2(96×96) → Dense(256) → BN → 256-dim
              ├──[LBP Branch]── multi-radius LBP → 256 bins → Dense(128) → 128-dim
              └──[HOG Branch]── HOG(12 orient) → 324-dim → Dense(128) → 128-dim
                          │
                 [Concatenate: 256+128+128 = 512-dim]
                          │
                 [Cross-Attention: Dense(512,sigmoid) × fused]
                          │
                 [ANFIS Projection: Dense(64,tanh) → LayerNorm]
                          │
              ┌───────────────────────────┐
              │        ANFIS CORE         │
              │ L1: Dual Fuzzification    │ ← Gaussian MF + Bell MF
              │ L2: Fuzzy Rule Layer      │ ← 48 rules, T-norm product
              │ L3: Normalization Layer   │ ← w̄_k = w_k/Σw_j
              │ L4: Consequent TSK        │ ← einsum, L2 reg
              │ L5: LayerNorm + GELU      │
              └───────────────────────────┘
                          │
              [Residual: Add + LayerNorm + GELU]
                          │
              ┌───────────┴────────────┐
              │   Emotion Head         │    Stress Head
              │ Dense(64→32→7,softmax) │  Dense(64→32→1,sigmoid×100)
              └────────────────────────┘
        ```
        """)

else:
    # Placeholder when no image
    st.markdown("""
    <div style='
        background: rgba(99,179,237,0.05);
        border: 2px dashed rgba(99,179,237,0.2);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 1rem;
    '>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>🖼️</div>
        <div style='color: #63b3ed; font-size: 1.1rem; font-weight: 600;'>
            Upload atau capture gambar wajah untuk memulai analisis
        </div>
        <div style='color: #718096; font-size: 0.9rem; margin-top: 0.5rem;'>
            Sistem akan otomatis menjalankan preprocessing pipeline dan inferensi ANFIS Hybrid
        </div>
        <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;'>
            <div style='color: #a0aec0; font-size: 0.85rem;'>✅ Upload JPG/PNG</div>
            <div style='color: #a0aec0; font-size: 0.85rem;'>✅ Capture Webcam</div>
            <div style='color: #a0aec0; font-size: 0.85rem;'>✅ Real-time Inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.8rem; padding: 0.5rem 0;'>
    ANFIS Hybrid Facial Emotion & Stress Analyzer &nbsp;|&nbsp;
    Soft Computing Project &nbsp;|&nbsp;
    Tim: Clarisya · Ica · Nazwa &nbsp;|&nbsp;
    Dataset: FER2013 (Kaggle) — 35.887 images
</div>
""", unsafe_allow_html=True)