import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Custom function to handle the model's specific preprocessing
@tf.keras.utils.register_keras_serializable()
def preprocess_input(x):
    return (x / 127.5) - 1.0

# 2. App Interface Setup
st.set_page_config(page_title="Solar Soiling Detector", layout="centered")
st.title("☀️ Solar Panel Soiling Detection")
st.write("Upload a photo of the solar panel to check its condition.")

# 3. Load the model from your file
@st.cache_resource
def load_my_model():
    model_path = 'soiling_detection_model.h5'
    # We pass the custom function so the app doesn't crash
    custom_objects = {'preprocess_input': preprocess_input}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

model = load_my_model()

# 4. Image Upload Button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("🔄 Analyzing...")

    # 5. Image Processing (Same as your Colab steps)
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (224, 224))
    img_final = preprocess_input(img_resized.astype(np.float32))
    img_final = np.expand_dims(img_final, axis=0)

    # 6. Make Prediction
    predictions = model.predict(img_final, verbose=0)
    class_labels = ['Clean', 'Heavily Soiled', 'Lightly Soiled']
    result = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # 7. Display Results
    st.divider()
    st.subheader(f"Result: {result}")
    st.progress(int(confidence))
    st.write(f"Confidence: {confidence:.2f}%")

    if result == 'Clean':
        st.success("The panel is clean! No maintenance needed.")
    else:
        st.error("Maintenance Alert: Cleaning is recommended to improve efficiency.")