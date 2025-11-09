import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pyttsx3
import time

SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40

# Cargar modelo y etiquetas
model = tf.keras.models.load_model("modelo_escucha_emocional.h5")
classes = np.load("label_classes.npy")
encoder = LabelEncoder()
encoder.classes_ = classes

# Inicializar TTS
tts = pyttsx3.init()

def decir(texto):
    print("[TTS]", texto)
    tts.say(texto)
    tts.runAndWait()

def grabar_audio(nombre="grabacion.wav"):
    print("🎙️ Grabando durante 3 segundos...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    write(nombre, SAMPLE_RATE, audio)
    print("✅ Grabación guardada como", nombre)
    return nombre

def procesar_audio(ruta):
    y, sr = librosa.load(ruta, sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc[np.newaxis, ..., np.newaxis]

def predecir_emocion():
    ruta = grabar_audio()
    X = procesar_audio(ruta)
    proba = model.predict(X)[0]
    idx = np.argmax(proba)
    emocion = classes[idx]
    confianza = proba[idx]

    print(f"Emoción detectada: {emocion} (confianza: {confianza:.2f})")

    frases = {
        "angry": "Parece que estás enojado.",
        "disgust": "Parece que algo te desagrada.",
        "fear": "Parece que estás asustado.",
        "happy": "Parece que estás feliz.",
        "neutral": "Tu voz suena bastante neutral.",
        "sad": "Parece que estás triste.",
        "calm": "Tu voz suena tranquila.",
        "surprise": "Parece que estás sorprendido."
    }
    decir(frases.get(emocion, f"Detecto la emoción {emocion}."))

if __name__ == "__main__":
    while True:
        predecir_emocion()
        cont = input("¿Grabar otra muestra? (s/n): ")
        if cont.lower() != 's':
            break
