import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# === 1. Cargar los datos ===
X = np.load("X_mfcc.npy")
y = np.load("y_labels.npy")

# Añadir un canal para CNN
X = X[..., np.newaxis]

# Convertir etiquetas a números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Guardar para decodificar luego
np.save("label_classes.npy", encoder.classes_)

# === 2. Dividir datos ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Datos de entrenamiento:", X_train.shape)
print("Datos de validación:", X_val.shape)

# === 3. Definir modelo CNN ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(np.unique(y_encoded)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === 4. Entrenamiento ===
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# === 5. Guardar modelo entrenado ===
model.save("modelo_escucha_emocional.h5")
print("✅ Modelo guardado como modelo_escucha_emocional.h5")
