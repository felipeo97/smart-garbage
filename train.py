import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import os

BASE_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

train_ds = image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    labels = 'inferred',
    label_mode = 'int',
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

val_ds = image_dataset_from_directory(
    os.path.join(BASE_DIR, "val"),
    labels = 'inferred',
    label_mode = 'int',
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = False,
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SIZE+(3,),
    include_top = False,
    weights = 'imagenet'
)

base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

history = model.fit(train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=5)

##model.save("saved_model_basura")

model.save("model_basura.keras")
print("Modelo guardado en model_basura.h5 y saved_model_basura/")