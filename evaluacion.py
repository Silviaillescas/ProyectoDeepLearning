import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# 1. Cargar datos y modelo
X = np.load("X_mfcc.npy")
y = np.load("y_labels.npy")
X = X[..., np.newaxis]

model = tf.keras.models.load_model("modelo_escucha_emocional.h5")
classes = np.load("label_classes.npy")

# 2. Volver a codificar etiquetas a los mismos índices que usamos al entrenar
encoder = LabelEncoder()
encoder.classes_ = classes
y_encoded = encoder.transform(y)

# 3. Predicciones
y_pred_proba = model.predict(X, batch_size=32)
y_pred = np.argmax(y_pred_proba, axis=1)

# 4. Reporte de clasificación
print("=== Classification report ===")
print(classification_report(y_encoded, y_pred, target_names=classes))

# 5. Matriz de confusión
cm = confusion_matrix(y_encoded, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de confusión - Escucha emocional")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.show()
print("Matriz de confusión guardada como matriz_confusion.png")
