from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

MODEL_PATH = os.path.join(settings.BASE_DIR, "model_basura.keras")
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["blanca", "caneca", "negra", "verde"]

def preprocess_image_file(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224,224))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return arr

@csrf_exempt
def classify_image(request):
    if request.method == "POST" and 'image' in request.FILES:
        f = request.FILES['image']
        img_bytes = f.read()
        x = preprocess_image_file(img_bytes)
        preds = model.predict(x)[0]
        idx = int(preds.argmax())
        return JsonResponse({
            "label": CLASS_NAMES[idx],
            "probability": float(preds[idx]),
            "all": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        })
    return JsonResponse({"error" : "POST con filed 'image' requerido"}, status=400)

def index(request):
    return render(request, "classifier/index.html")