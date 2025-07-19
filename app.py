import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Load model
model = load_model("wastesort_model.h5")

# Class labels (must match training order)
class_names = ['Biodegradable', 'Hazardous', 'Recyclable']

st.title("♻️ WasteSort AI – Smart Waste Classifier")
st.markdown("Upload an image of any waste item and let AI classify it.")

# Upload
uploaded_file = st.file_uploader("Upload a waste image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

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
