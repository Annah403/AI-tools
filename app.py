# ==============================================================
# MNIST Digit Classifier Web App
# Using Streamlit
# ==============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------------------
# Load the trained model
# --------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

# --------------------------------------------------------------
# App UI
# --------------------------------------------------------------
st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9), and I'll predict what it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(image)
    img_array = 255 - img_array  # Invert (white background → black)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width=150)
    st.write("Processing...")

    # Predict digit
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.subheader(f"✅ Predicted Digit: {predicted_label}")

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
