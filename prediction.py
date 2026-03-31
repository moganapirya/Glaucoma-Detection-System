# SCRIPT 05: STREAMLIT DASHBOARD (app.py)
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Glaucoma Detection") 

@st.cache_resource # Optimized loading [cite: 1421]
def get_model():
    return tf.keras.models.load_model("glaucoma_model.h5")

model = get_model()

st.title("Glaucoma Detection Dashboard") 
uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png"]) 

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess [cite: 1428-1433]
    img_resized = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    
    # Predict [cite: 1442-1450]
    prob = model.predict(img_array)[0][0]
    label = "Glaucoma" if prob > 0.5 else "Normal"
    confidence = prob if prob > 0.5 else 1 - prob
    
    st.subheader(f"Result: {label}")
    st.write(f"Confidence: {confidence:.4f}")
    st.info("Educational use only. Not a clinical tool.") 