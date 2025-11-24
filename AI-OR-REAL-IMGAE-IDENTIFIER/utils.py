import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

# ----------------------------------------------------
# 1. DATA AUGMENTATION LAYER
# ----------------------------------------------------
def get_data_augmentation():
    """Returns a Sequential layer for image augmentation."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name="data_augmentation")
    return data_augmentation


# ----------------------------------------------------
# 2. IMAGE PREPROCESSING
# ----------------------------------------------------
def preprocess_image(image, img_size=(180, 180)):
    """Preprocess a single image (resize + normalize)."""
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


# ----------------------------------------------------
# 3. PLOT TRAINING HISTORY
# ----------------------------------------------------
def plot_training_history(history):
    """Plot training & validation accuracy and loss."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 4. COUNT IMAGES IN DATASET
# ----------------------------------------------------
def count_images_in_dataset(dataset):
    """Count total number of images in a TensorFlow dataset."""
    return sum(1 for _ in dataset.unbatch())


# ----------------------------------------------------
# 5. PREDICTION HELPER
# ----------------------------------------------------
def predict_image(model, image):
    """Predict whether an image is AI-generated or real."""
    img_array = preprocess_image(image)
    img_array = tf.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0][0]
    if pred < 0.5:
        return "AI-GENERATED", float((1 - pred) * 100)
    else:
        return "REAL", float(pred * 100)
