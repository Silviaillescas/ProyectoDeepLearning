# ProyectoDeepLearning
Michelle Mejía 22596 - Silvia Illescas 22376

🎧 Escucha Emocional

Escucha Emocional es un sistema de análisis de voz basado en Deep Learning que permite reconocer la emoción predominante en una grabación de audio. Utiliza una red neuronal convolucional (CNN) entrenada con coeficientes MFCC y una interfaz construida con Streamlit que permite grabar voz en tiempo real o cargar archivos .wav para su análisis.

📁 Estructura del proyecto
ProyectoDeepLearning/
│
├── datos/                        # Carpeta con audios base (no incluida en el repo por tamaño)
├── preprocesamiento.py           # Genera características MFCC y etiquetas
├── modelo_cnn.py                 # Define y entrena el modelo CNN
├── app_streamlit.py              # Interfaz interactiva en Streamlit
├── inferencia_tiempo_real.py     # Pruebas de inferencia con audios individuales
├── evaluacion.py                 # Cálculo de métricas y matriz de confusión
├── modelo_escucha_emocional.h5   # Modelo entrenado
├── label_classes.npy             # Clases de emociones codificadas
├── X_mfcc.npy, y_labels.npy      # Datos preprocesados
├── README.md                     # Este documento
└── .gitignore                    # Exclusión de archivos grandes

🧠 Arquitectura del modelo

El modelo CNN consta de tres bloques convolucionales seguidos de Batch Normalization y MaxPooling, una capa totalmente conectada de 128 neuronas y una capa de salida softmax con 8 clases de emociones:

Emociones reconocidas: angry, calm, disgust, fear, happy, neutral, sad, surprise

Optimización: Adam (lr = 0.001)

Función de pérdida: categorical crossentropy

Precisión alcanzada: ~89%

💻 Interfaz Streamlit

La aplicación incluye dos formas de análisis:

🎙️ Grabación directa: captura audio desde el micrófono y predice la emoción en tiempo real.

📁 Carga de archivo: permite subir un .wav y visualizar la distribución de probabilidades.

Cada predicción incluye:

La emoción principal detectada y su confianza.

Una gráfica de barras con las probabilidades por emoción.

Un control de umbral ajustable.

La opción de descargar los resultados en formato .csv.

🔍 Validación del modelo

La validación se realizó tanto con el conjunto de datos reservado como con audios de usuarios reales.
El modelo mostró mayor precisión en las emociones happy, sad y angry, mientras que fear y disgust presentaron confusión parcial.
Se observó que las probabilidades cambian debido al uso de una capa softmax, la cual distribuye el 100 % de la confianza entre las clases.
En trabajos futuros se propone un enfoque multi-etiqueta con activaciones sigmoid para detectar emociones simultáneas.

⚙️ Instalación y ejecución
1️⃣ Requisitos previos

Python 3.9 o superior

TensorFlow 2.19+

Streamlit 1.50+

2️⃣ Instalación de dependencias
pip install -r requirements.txt

3️⃣ Ejecución del proyecto
# Preprocesar los datos
python preprocesamiento.py

# Entrenar el modelo
python modelo_cnn.py

# Ejecutar la interfaz
python -m streamlit run app_streamlit.py

🧩 Resultados esperados

Exactitud global: 0.89

Macro F1-score: 0.90

Matriz de confusión:
Las emociones angry, happy y sad mostraron los mejores niveles de precisión.
