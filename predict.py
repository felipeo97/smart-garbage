import tensorflow as tf
import numpy as np
from PIL import Image
import sys

CLASS_NAMES = ["blanca", "caneca", "negra", "verde"]

model = tf.keras.models.load_model("model_basura.keras")

def predict(path):
    img = Image.open(path).convert("RGB").resize((224,224))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    idx = preds.argmax()
    return CLASS_NAMES[idx], float(preds[idx])

if __name__ == "__main__":
    image_path = sys.argv[1]
    label, prob = predict(image_path)
    print(label, prob)
