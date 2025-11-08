import os
import numpy as np
import librosa

# === CONFIGURACIÓN GENERAL ===
CREMAD_DIR = r"C:\Users\Silvia\Documents\Cuarto Año\Segundo Semestre\Deeplearning\ProyectoDeepLearning\datos\AudioWAV"
RAVDESS_DIR = r"C:\Users\Silvia\Documents\Cuarto Año\Segundo Semestre\Deeplearning\ProyectoDeepLearning\datos"
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40

# === MAPEOS DE EMOCIONES ===
CREMA_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

RAVDESS_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

def load_audio(file_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    y, sr = librosa.load(file_path, sr=sample_rate)
    target_len = int(sample_rate * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))))
    return y, sr

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc

def process_cremad(data_dir):
    X, y = [], []
    for f in os.listdir(data_dir):
        if f.endswith(".wav"):
            emo_code = f.split("_")[2]
            label = CREMA_MAP.get(emo_code)
            if label:
                path = os.path.join(data_dir, f)
                y_audio, sr = load_audio(path)
                mfcc = extract_mfcc(y_audio, sr)
                X.append(mfcc)
                y.append(label)
    return X, y

def process_ravdess(base_dir):
    X, y = [], []
    for folder in os.listdir(base_dir):
        if not folder.startswith("Actor_"):
            continue
        folder_path = os.path.join(base_dir, folder)
        for f in os.listdir(folder_path):
            if f.endswith(".wav"):
                emo_code = f.split("-")[2]
                label = RAVDESS_MAP.get(emo_code)
                if label:
                    path = os.path.join(folder_path, f)
                    y_audio, sr = load_audio(path)
                    mfcc = extract_mfcc(y_audio, sr)
                    X.append(mfcc)
                    y.append(label)
    return X, y

def build_dataset():
    X_crema, y_crema = process_cremad(CREMAD_DIR)
    X_rav, y_rav = process_ravdess(RAVDESS_DIR)

    X = np.array(X_crema + X_rav)
    y = np.array(y_crema + y_rav)

    print("MFCCs shape:", X.shape)
    print("Labels shape:", y.shape)
    np.save("X_mfcc.npy", X)
    np.save("y_labels.npy", y)
    print("Guardado X_mfcc.npy y y_labels.npy")

if __name__ == "__main__":
    build_dataset()
