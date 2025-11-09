import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os

# =========================
# Configuración del modelo
# =========================
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40

MODEL_PATH = "modelo_escucha_emocional.h5"
LABELS_PATH = "label_classes.npy"

@st.cache_resource
def cargar_modelo_y_clases():
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(LABELS_PATH)
    encoder = LabelEncoder()
    encoder.classes_ = classes
    return model, classes, encoder

model, classes, encoder = cargar_modelo_y_clases()

# =========================
# Funciones de audio
# =========================
def load_audio_fixed(path_or_file, sample_rate=SAMPLE_RATE, duration=DURATION):
    """
    Carga audio desde ruta o file-like y lo ajusta a duración fija.
    """
    y, sr = librosa.load(path_or_file, sr=sample_rate)
    target_len = int(sample_rate * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))))
    return y, sr

def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def grabar_audio_streamlit(duracion=DURATION, sample_rate=SAMPLE_RATE):
    """
    Graba desde el micrófono usando sounddevice y devuelve la ruta a un .wav temporal.
    """
    st.info(f"Grabando {duracion} segundos... habla ahora 🗣️")
    audio = sd.rec(int(duracion * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    # Guardar a un archivo temporal
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmpfile.name, sample_rate, audio)
    st.success("Grabación completa ✅")
    return tmpfile.name

def predecir_emocion_desde_path(file_path):
    y_audio, sr = load_audio_fixed(file_path)
    mfcc = extract_mfcc(y_audio, sr)      # (n_mfcc, tiempo)
    X = mfcc[np.newaxis, ..., np.newaxis] # (1, n_mfcc, tiempo, 1)

    proba = model.predict(X)[0]
    idx = int(np.argmax(proba))
    emocion = classes[idx]
    confianza = float(proba[idx])

    proba_dict = {clase: float(p) for clase, p in zip(classes, proba)}

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
    texto = f"**Emoción detectada:** `{emocion}`  \n" \
            f"**Confianza:** `{confianza:.2f}`  \n\n" \
            f"{frases.get(emocion, f'Se detectó la emoción {emocion}.')}"

    return texto, proba_dict

# =========================
# Interfaz Streamlit
# =========================
st.set_page_config(page_title="Escucha Emocional", page_icon="🎧", layout="centered")

st.title("🎧 Escucha Emocional")
st.markdown(
    """
    Analizador de emociones en voz usando Deep Learning.  
    Puedes **grabar tu voz** o **subir un archivo .wav** y el modelo estimará la emoción principal.
    """
)

tab1, tab2 = st.tabs(["🎙️ Grabar desde micrófono", "📁 Subir archivo de audio"])

with tab1:
    st.subheader("Grabar voz")
    st.write("Haz clic en el botón, espera a que termine la grabación y luego se mostrará la predicción.")

    if st.button("Grabar 3 segundos"):
        try:
            tmp_path = grabar_audio_streamlit()
            # Mostrar el audio grabado
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")

            texto, proba_dict = predecir_emocion_desde_path(tmp_path)

            st.markdown("### Resultado")
            st.markdown(texto)
            st.markdown("### Probabilidades por emoción")
            st.bar_chart(proba_dict)

            # limpiar archivo temporal
            os.remove(tmp_path)
        except Exception as e:
            st.error(f"Ocurrió un error al grabar o procesar el audio: {e}")

with tab2:
    st.subheader("Subir archivo .wav")
    archivo = st.file_uploader("Selecciona un archivo de audio (.wav)", type=["wav"])

    if archivo is not None:
        # Guardar a archivo temporal para que librosa lo pueda leer
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmpfile.write(archivo.read())
        tmpfile.flush()

        st.audio(tmpfile.name, format="audio/wav")

        if st.button("Analizar este audio"):
            try:
                texto, proba_dict = predecir_emocion_desde_path(tmpfile.name)
                st.markdown("### Resultado")
                st.markdown(texto)
                st.markdown("### Probabilidades por emoción")
                st.bar_chart(proba_dict)
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo: {e}")
            finally:
                os.remove(tmpfile.name)
