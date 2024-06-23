#app

from typing import Any
import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import activations
from keras import backend as K
from PIL import Image
from keras.models import load_model
import glob 

st.title("Alzheimer's Disease Predictor")
st.write("This app takes in MRI images of different patients and uses machine learning (AI) to predict whether the patient has Alzheimer's or not. This model also determines the potential severity.")
st.caption(":red[Note: This app is not meant to act as a definitive diagnostic tool.]")

model = keras.models.load_model("C:\\Users\\mrkit\\Documents\\Streamlit\\my_model.h5")

img = st.file_uploader("Upload your image here...", type=["png", "jpg"])
if img is not None:
    image = st.image(img, caption="Your image")
    pilimage = Image.open(img).convert("RGB")
    data = np.array(pilimage)
    def normalize_image(pilimage):
        
        return (pilimage - np.min(pilimage)) / (np.max(pilimage) - np.min(pilimage))

    normalized = []
    for img in data:
        normalized.append(normalize_image(img))
    
    normalized = np.array(normalized)
    normalized = normalized.reshape(1, 128, 128, 3)
    normalized = normalized.astype('float32')
    normalized /= 255


if st.button("Go!"):
    result = model.predict(normalized)
    prediction = np.argmax(result)
    st.write("Prediction chart:", result)
    if prediction == 0:
        st.write("This MRI scan has no signs of Alzheimer's.")
    elif prediction == 1:
        st.write("This MRI scan has very mild signs of Alzheimer's.")
    elif prediction == 2:
        st.write("This MRI scan has mild signs of Alzheimer's.")
    elif prediction == 3:
        st.write("This MRI scan has moderate to severe signs of Alzheimer's.")


