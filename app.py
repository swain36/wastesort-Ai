import streamlit as st
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Google Drive .h5 model URL (direct download)
MODEL_URL = "https://drive.google.com/uc?export=download&id=130sEJBg48KqmY8-sbxjgtY9XS_KKY2SO"
MODEL_PATH = "wastesort_model.h5"

# Download and load model
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return load_model(MODEL_PATH)

# Load the model
model = get_model()

# Class labels (must match training order)
class_names = ['Biodegradable', 'Hazardous', 'Recyclable']

# UI
st.title("♻️ WasteSort AI – Smart Waste Classifier")
st.markdown("Upload an image of any waste item and let AI classify it.")

# File uploader
uploaded_file = st.file_uploader("Upload a waste image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_label = class_names[class_index]

    st.markdown(f"### 🧠 Prediction: **{predicted_label}**")

    # Suggestion
    suggestions = {
        "Biodegradable": "🌱 Compost this or dispose in green bins.",
        "Hazardous": "⚠️ Dispose at hazardous collection centers.",
        "Recyclable": "🔁 Clean and recycle via dry waste programs."
    }
    st.info(suggestions[predicted_label])
