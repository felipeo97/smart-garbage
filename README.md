# 🧠 Basurero Inteligente con Django y Python

Este proyecto es un **sistema de clasificación inteligente de residuos**.  
Permite subir una imagen de basura y el sistema identifica automáticamente si pertenece a la caneca **verde**, **negra** o **blanca**, usando un modelo de **inteligencia artificial entrenado con TensorFlow/Keras**.

El proyecto está hecho con **Django (backend y frontend web)** y **Python** para el procesamiento de imágenes y el entrenamiento del modelo.

---

## 🚀 Características

- Web sencilla (una sola página) desarrollada en Django.  
- Subida de imágenes desde el navegador.  
- Clasificación de imágenes en tiempo real (verde / negra / blanca).  
- Modelo de IA entrenado localmente con TensorFlow y Keras.  
- Código simple y educativo para entender el flujo completo.

## Esctructura del proyecto
basurero-inteligente/
│
├── manage.py
├── train.py # Script para entrenar el modelo
├── model_basura.h5 # Modelo entrenado (se genera al correr train.py)
│
├── core/ # Configuración principal de Django
│ ├── settings.py
│ ├── urls.py
│ └── ...
│
├── classifier/ # App principal
│ ├── views.py # Contiene la vista que recibe la imagen
│ ├── urls.py
│ ├── templates/
│ │ └── index.html # Página principal del proyecto
│ └── static/
│ └── styles.css # (opcional)
│
└── dataset/ # Imágenes usadas para el entrenamiento
├── train/
│ ├── verde/
│ ├── negra/
│ └── blanca/
└── val/
├── verde/
├── negra/
└── blanca/

## Crear entorno virtual
Ejecutar los siguientes comandos en este orden
python -m venv .venv
source .venv/bin/activate       # En Linux/Mac
.venv\Scripts\activate          # En Windows
pip install django tensorflow pillow numpy

## Ejecutar servidor DJango
python manage.py runserver
