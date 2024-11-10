import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("edge_detection.h5", compile=False)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, 0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# Function to predict the class of the image
def predict_image_class(image, model, threshold=0.5):
    image = preprocess_image(image)
    predictions = model.predict(image)
    score = predictions.squeeze()
    if score >= threshold:
        return f"This image is {100 * score:.2f}% malignant."
    else:
        return f"This image is {100 * (1 - score):.2f}% benign."

# Streamlit app
st.title("Skin Cancer Detection")
st.write("Upload an image of a skin lesion to classify it as benign or malignant.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image_class(image, model)
    st.write(prediction)
