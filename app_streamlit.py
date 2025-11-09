import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import time
import plotly.express as px

# =========================
# CONFIGURACIÓN DEL MODELO
# =========================
SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40
MODEL_PATH = "modelo_escucha_emocional.h5"
LABELS_PATH = "label_classes.npy"

# =========================
# CARGAR MODELO Y CLASES
# =========================
@st.cache_resource
def cargar_modelo_y_clases():
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(LABELS_PATH)
    encoder = LabelEncoder()
    encoder.classes_ = classes
    return model, classes, encoder

model, classes, encoder = cargar_modelo_y_clases()

# =========================
# FUNCIONES DE AUDIO
# =========================
def load_audio_fixed(path_or_file, sample_rate=SAMPLE_RATE, duration=DURATION):
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
    st.info(f"🎙️ Grabando {duracion} segundos... habla ahora")
    audio = sd.rec(int(duracion * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmpfile.name, sample_rate, audio)
    st.success("✅ Grabación completa")
    return tmpfile.name

def predecir_emocion_desde_path(file_path, umbral=0.2):
    y_audio, sr = load_audio_fixed(file_path)
    mfcc = extract_mfcc(y_audio, sr)
    X = mfcc[np.newaxis, ..., np.newaxis]

    proba = model.predict(X)[0]
    proba_dict = {clase: float(p) for clase, p in zip(classes, proba)}

    # Emoción principal
    idx = int(np.argmax(proba))
    emocion_principal = classes[idx]
    confianza_principal = float(proba[idx])

    # DataFrame con todas las emociones
    df = pd.DataFrame({
        "emocion": list(classes),
        "probabilidad": [proba_dict[c] for c in classes]
    })

    # Marcar emociones activas según umbral
    df["es_emocion_predicha"] = df["probabilidad"] >= umbral
    emociones_activas = df[df["es_emocion_predicha"]]["emocion"].tolist()

    frases = {
        "angry": "😠 Tu voz refleja enojo o frustración.",
        "disgust": "😖 Se percibe desagrado o disgusto.",
        "fear": "😨 Tu tono sugiere miedo o ansiedad.",
        "happy": "😊 Se detecta felicidad y energía positiva.",
        "neutral": "😐 Tu voz suena tranquila y neutral.",
        "sad": "😢 Se nota tristeza o melancolía.",
        "calm": "😌 Tu voz es serena y relajada.",
        "surprise": "😲 Suena como si algo te sorprendiera."
    }

    texto = (
        f"### 🎧 Emoción principal: **{emocion_principal.upper()}**\n"
        f"Confianza: `{confianza_principal:.2f}`\n\n"
        f"Emociones por encima del umbral ({umbral:.2f}): "
        f"`{', '.join(emociones_activas) if emociones_activas else 'Ninguna'}`\n\n"
        f"{frases.get(emocion_principal, f'Se detectó la emoción {emocion_principal}.')}"
    )

    return texto, df

# =========================
# INTERFAZ STREAMLIT
# =========================
st.set_page_config(page_title="Escucha Emocional", page_icon="🎧", layout="centered")

st.title("🎧 Escucha Emocional")
st.markdown(
    """
    Analizador de **emociones en la voz** usando redes neuronales convolucionales.  
    Puedes **grabar tu voz** o **subir un archivo .wav**, y el modelo estimará la emoción principal.
    """
)

# Slider de umbral en la barra lateral
umbral = st.sidebar.slider(
    "Umbral para marcar emociones como activas",
    min_value=0.0, max_value=1.0, value=0.2, step=0.05
)

tab1, tab2 = st.tabs(["🎙️ Grabar voz", "📁 Subir archivo de audio"])

# === TAB 1: Grabación ===
with tab1:
    st.subheader("Grabar desde micrófono")
    st.write("Haz clic en el botón, espera a que termine la grabación y luego se mostrará la predicción.")

    if st.button("🎤 Iniciar grabación de 3 segundos"):
        tmp_path = None
        try:
            tmp_path = grabar_audio_streamlit()
            with open(tmp_path, "rb") as f:
                st.audio(f.read(), format="audio/wav")

            texto, df_result = predecir_emocion_desde_path(tmp_path, umbral=umbral)
            st.markdown(texto)

            fig = px.bar(
                df_result,
                x="emocion",
                y="probabilidad",
                color="es_emocion_predicha",
                color_discrete_map={True: "#FF6F91", False: "#9AD0EC"},
                labels={
                    "emocion": "Emoción",
                    "probabilidad": "Probabilidad",
                    "es_emocion_predicha": "≥ umbral"
                },
                title="Distribución de probabilidades por emoción"
            )
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar resultados (CSV)",
                data=csv_bytes,
                file_name="resultado_escucha_emocional.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
        finally:
            if tmp_path is not None:
                try:
                    time.sleep(1)
                    os.remove(tmp_path)
                except PermissionError:
                    pass

# === TAB 2: Subir archivo ===
with tab2:
    st.subheader("Analizar archivo de audio")
    archivo = st.file_uploader("Selecciona un archivo .wav", type=["wav"])

    if archivo is not None:
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmpfile.write(archivo.read())
        tmpfile.flush()

        st.audio(tmpfile.name, format="audio/wav")

        if st.button("🔍 Analizar audio"):
            try:
                texto, df_result = predecir_emocion_desde_path(tmpfile.name, umbral=umbral)
                st.markdown(texto)

                fig = px.bar(
                    df_result,
                    x="emocion",
                    y="probabilidad",
                    color="es_emocion_predicha",
                    color_discrete_map={True: "#FF6F91", False: "#9AD0EC"},
                    labels={
                        "emocion": "Emoción",
                        "probabilidad": "Probabilidad",
                        "es_emocion_predicha": "≥ umbral"
                    },
                    title="Distribución de probabilidades por emoción"
                )
                fig.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)

                csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar resultados (CSV)",
                    data=csv_bytes,
                    file_name="resultado_escucha_emocional.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
            finally:
                try:
                    time.sleep(1)
                    os.remove(tmpfile.name)
                except PermissionError:
                    pass

st.divider()
st.caption("Proyecto Deep Learning 2025")
