import os
import csv

# Rutas a tus datos (ajusta si cambian)
CREMAD_DIR = r"C:\Users\Silvia\Documents\Cuarto Año\Segundo Semestre\Deeplearning\ProyectoDeepLearning\datos\AudioWAV"
RAVDESS_DIR = r"C:\Users\Silvia\Documents\Cuarto Año\Segundo Semestre\Deeplearning\ProyectoDeepLearning\datos"

# Mapas de emociones CREMA-D (códigos del nombre de archivo)
CREMA_MAP_EN = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}
CREMA_MAP_ES = {
    "ANG": "enojo",
    "DIS": "asco",
    "FEA": "miedo",
    "HAP": "felicidad",
    "NEU": "neutral",
    "SAD": "tristeza",
}

# Mapas de emociones RAVDESS (tercer bloque del nombre: 01–08)
RAVDESS_MAP_EN = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
RAVDESS_MAP_ES = {
    "01": "neutral",
    "02": "calma",
    "03": "felicidad",
    "04": "tristeza",
    "05": "enojo",
    "06": "miedo",
    "07": "asco",
    "08": "sorpresa",
}

salida_csv = "etiquetas_audios.csv"

filas = []

# ---- CREMA-D ----
for fname in os.listdir(CREMAD_DIR):
    if not fname.lower().endswith(".wav"):
        continue
    parts = fname.split("_")
    if len(parts) < 3:
        continue
    emo_code = parts[2]
    emo_en = CREMA_MAP_EN.get(emo_code)
    emo_es = CREMA_MAP_ES.get(emo_code)
    if emo_en is None:
        continue

    ruta_rel = os.path.join("datos", "AudioWAV", fname)
    filas.append([
        "CREMA-D",
        ruta_rel,
        fname,
        emo_en,
        emo_es,
    ])

# ---- RAVDESS ----
for folder in os.listdir(RAVDESS_DIR):
    if not folder.startswith("Actor_"):
        continue
    actor_path = os.path.join(RAVDESS_DIR, folder)
    if not os.path.isdir(actor_path):
        continue

    for fname in os.listdir(actor_path):
        if not fname.lower().endswith(".wav"):
            continue
        parts = fname.split("-")
        if len(parts) < 3:
            continue
        emo_code = parts[2]
        emo_en = RAVDESS_MAP_EN.get(emo_code)
        emo_es = RAVDESS_MAP_ES.get(emo_code)
        if emo_en is None:
            continue

        ruta_rel = os.path.join("datos", folder, fname)
        filas.append([
            "RAVDESS",
            ruta_rel,
            fname,
            emo_en,
            emo_es,
        ])

# ---- Guardar CSV ----
with open(salida_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "ruta_relativa", "archivo", "emocion_en", "emocion_es"])
    writer.writerows(filas)

print(f"✅ Se generó el archivo {salida_csv} con {len(filas)} audios etiquetados.")
