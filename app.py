import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fc_vgg16.keras')
    return model

model = load_model()

st.write("""
# Flower Classification System
""")
file = st.file_uploader("Choose a flower photo from your computer: daisy, dandelion, sunflower, rose, or tulip.", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    result = f"OUTPUT: {class_names[np.argmax(prediction)]}"
    st.success(result)
