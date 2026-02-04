from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
from pathlib import Path
import numpy as np
import joblib
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")  # <-- Add this at the very top, before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import shap
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from lime import lime_tabular
from sklearn.inspection import permutation_importance
import seaborn as sns


# ------------------ Load models ------------------
ann_loaded = keras.models.load_model("saved_models/ann_model.h5")
scaler_loaded = joblib.load("saved_models/scaler.pkl")
encoder_loaded = joblib.load("saved_models/label_encoder.pkl")

# ------------------ Flask app ------------------
app = Flask(__name__)
CORS(app)

TRAIN_DIR = Path("./mgc_split/train")
TEST_DIR  = Path("./mgc_split/test")


# ------------------ Constants ------------------
SAMPLE_RATE = 22050
N_MFCC = 40
DURATION = 45
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

FEATURE_NAMES = [f"mfcc_mean_{i}" for i in range(N_MFCC)] + [f"mfcc_std_{i}" for i in range(N_MFCC)]

# ------------------ Feature extraction ------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    # if len(y) < SAMPLES_PER_TRACK:
    #     y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    # else:
    #     y = y[:SAMPLES_PER_TRACK]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])
    return features

def load_dataset(folder_path):
    X, y = [], []
    for class_dir in folder_path.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for file in class_dir.glob("*.mp3"):
                feat = extract_features(file)
                X.append(feat)
                y.append(label)
            for file in class_dir.glob("*.wav"):
                feat = extract_features(file)
                X.append(feat)
                y.append(label)
    return np.array(X), np.array(y)

def get_scaled_feature(file_path):
    feat = extract_features(file_path).reshape(1, -1)
    return scaler_loaded.transform(feat)

# --- STEP 5: Load datasets ---
X_train, y_train = load_dataset(TRAIN_DIR)
X_test, y_test   = load_dataset(TEST_DIR)

# Encode labels
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# Split train into train + validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc
)

def compute_permutation_importance(model, X, y, n_repeats=5):
    """Compute permutation importance and return sorted indices and mean importances"""
    # Wrapper for ANN
    def ann_predict(X_input):
        return np.argmax(model.predict(X_input), axis=1)

    if isinstance(model, keras.Model):
        scoring_func = lambda est, X, y: np.mean(ann_predict(X) == y)
    else:
        scoring_func = None  # default accuracy for sklearn models

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring=scoring_func
    )

    sorted_idx = result.importances_mean.argsort()[::-1]
    return result, sorted_idx

from pydub import AudioSegment
from pydub.utils import which

# Ensure FFmpeg path is set (important for venv)
AudioSegment.converter = which("ffmpeg") or "C:\\ffmpeg\\bin\\ffmpeg.exe"

def convert_to_mp3_22050_mono(input_path, output_path="temp_audio.mp3"):
    """
    Converts any audio file to MP3 format, mono channel, 22050 Hz sampling rate.
    Saves as temp_audio.mp3 in the same directory.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(output_path, format="mp3")
        print(f"✅ Converted {input_path} → {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return None

# ------------------ ANN top-3 prediction ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Save the uploaded file temporarily with its original extension
    original_filename = file.filename
    temp_input_path = f"temp_input{Path(original_filename).suffix}"  # e.g., temp_input.wav, temp_input.m4a
    file.save(temp_input_path)

    # Convert uploaded audio to standard MP3, mono, 22050 Hz
    converted_path = convert_to_mp3_22050_mono(temp_input_path)  # will be saved as temp_audio.mp3
    if not converted_path:
        return jsonify({"error": "Audio conversion failed"}), 500

    feat_scaled = get_scaled_feature(converted_path)
    probs = ann_loaded.predict(feat_scaled)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_genres = encoder_loaded.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    results = [{"genre": str(g), "probability": float(p)} for g, p in zip(top3_genres, top3_probs)]
    return jsonify({"ann_top3": results})

@app.route('/explain/lime', methods=['POST'])
def explain_lime():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Save the uploaded file temporarily with its original extension
    original_filename = file.filename
    temp_input_path = f"temp_input{Path(original_filename).suffix}"  # e.g., temp_input.wav, temp_input.m4a
    file.save(temp_input_path)

    # Convert uploaded audio to standard MP3, mono, 22050 Hz
    converted_path = convert_to_mp3_22050_mono(temp_input_path)  # will be saved as temp_audio.mp3
    if not converted_path:
        return jsonify({"error": "Audio conversion failed"}), 500

    # Extract and scale features
    feat = extract_features(converted_path).reshape(1, -1)
    feat_scaled = scaler_loaded.transform(feat)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_final,
        feature_names=[f"MFCC_{i}" for i in range(X_train_final.shape[1])],
        class_names=encoder_loaded.classes_,
        mode='classification',
    )

    def ann_predict_proba(x):
        return ann_loaded.predict(x)

    predicted_class_idx = np.argmax(ann_predict_proba(feat_scaled)[0])
    explanation = explainer.explain_instance(
        data_row=feat_scaled[0],
        predict_fn=ann_predict_proba,
        top_labels=1,
        labels=(predicted_class_idx,)
    )

    lime_html = explanation.as_html()
    return jsonify({"lime_html": lime_html})


# ------------------ SHAP explanation ------------------
@app.route('/explain/shap', methods=['POST'])
def explain_shap():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Save the uploaded file temporarily with its original extension
    original_filename = file.filename
    temp_input_path = f"temp_input{Path(original_filename).suffix}"  # e.g., temp_input.wav, temp_input.m4a
    file.save(temp_input_path)

    # Convert uploaded audio to standard MP3, mono, 22050 Hz
    converted_path = convert_to_mp3_22050_mono(temp_input_path)  # will be saved as temp_audio.mp3
    if not converted_path:
        return jsonify({"error": "Audio conversion failed"}), 500


    feat = extract_features(converted_path).reshape(1, -1)
    feat_scaled = scaler_loaded.transform(feat)

    background = X_train_final[:100]
    explainer = shap.KernelExplainer(ann_loaded.predict, background)
    shap_values = explainer.shap_values(feat_scaled, nsamples=50)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feat_scaled,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        class_names=list(encoder_loaded.classes_),
        show=False
    )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return jsonify({"shap_image": img_base64})

@app.route('/explain/permutation', methods=['POST'])
def explain_permutation():
    # Use the test set for explanation
    X_used = X_test
    y_used = y_test_enc

    # Compute permutation importance
    result, sorted_idx = compute_permutation_importance(ann_loaded, X_used, y_used)

    # Feature names
    mfcc_features = [f"MFCC{i+1}_mean" for i in range(N_MFCC)] + [f"MFCC{i+1}_std" for i in range(N_MFCC)]

    # ---- Bar plot ----
    plt.figure(figsize=(20, 16))
    plt.barh(
        np.array(mfcc_features)[sorted_idx][::-1],
        result.importances_mean[sorted_idx][::-1]
    )
    plt.xlabel("Mean Importance Decrease")
    plt.title("Permutation Feature Importance (MFCC Features)")
    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()


    return jsonify({
        "bar_plot": img_base64
    })

if __name__ == '__main__':
    print('\033[92m✔Backend connected and running\033[0m \n')
    app.run(debug=True)