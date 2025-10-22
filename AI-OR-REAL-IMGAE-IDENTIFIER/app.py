import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import os

@st.cache_resource
def load_model_app():
    # Get the directory where this file is located
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, "model_weights.h5")

    # Define the model structure
    base_model = MobileNetV2(input_shape=(180, 180, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    # Load weights safely
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        st.error(f"Model weights file not found at: {weights_path}")
    return model

model = load_model_app()

st.title("🧠 AI or Real Image Identifier")
st.write("Upload an image to determine if it is AI-generated or real.")

upload_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if upload_file:
    image = Image.open(upload_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(image.resize((180, 180))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]
    if pred < 0.5:
        result = "🤖 AI-GENERATED"
        confidence = (1 - pred) * 100
        description = "This image is AI-generated using artificial techniques."
    else:
        result = "📷 REAL"
        confidence = pred * 100
        description = "This image appears to be real and naturally captured."

    st.subheader(f"Prediction Result: {result}")
    st.write(description)
    st.write(f"Confidence: {confidence:.2f}%")
