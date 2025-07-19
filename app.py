import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
import os
import gdown

MODEL_PATH = "wastesort_model.h5"
GDRIVE_FILE_ID = "130sEJBg48KqmY8-sbxjgtY9XS_KKY2SO"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
class_names = ['Biodegradable', 'Hazardous', 'Recyclable']

st.title("♻️ WasteSort AI – Smart Waste Classifier")
st.markdown("Upload an image of any waste item and let AI classify it.")

uploaded_file = st.file_uploader("Upload a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_label = class_names[class_index]

    st.markdown(f"### 🧠 Prediction: **{predicted_label}**")

    suggestions = {
        "Biodegradable": "🌱 Compost this or dispose in green bins.",
        "Hazardous": "⚠️ Dispose at hazardous collection centers.",
        "Recyclable": "🔁 Clean and recycle via dry waste programs."
    }
    st.info(suggestions[predicted_label])
