import streamlit as st
import PIL
from PIL import Image

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

from streamlit.cli import main


st.title("Dog Breed Identification")

#image_app = Image.open('Prediction\Lowchen-Toy-Dog-Breed.jpg')
#st.image(image_app)

#picture = st.camera_input()
picture = st.file_uploader("Choose a file")

try:
    image = Image.open(picture)
    st.image(image)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1


    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Load the image into the array
    data[0] = normalized_image_array
    model = load_model('keras_model.h5')

    prediction = model.predict(data)

    with open('labels.txt') as label:
        labels = label.readlines()
        st.write('labels')

    for idx, val in enumerate(labels) :
        labels[idx] = val.strip()[2:].strip()

    results = pd.DataFrame({'labels':labels, 'proba':prediction[0]})
    results.sort_values(by = 'proba', inplace = True, ascending= False)

    s = 0
    i = 0
    while s <= 0.99:
        st.write('Il y a ', str(round(100 * results.iloc[i, 1], 1))+'%','de chances que ce chien soit un',  results.iloc[i, 0])
        s += results.iloc[i, 1]
        i = i+1
except:
    st.write("Pas d'image detectée")