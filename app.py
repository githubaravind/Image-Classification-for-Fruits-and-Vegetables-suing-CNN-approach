import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🥦 Fruit & Vegetable Classifier")

# ✅ Load model (keep model in same folder OR use correct path)
model = load_model(r'D:\CV_project\Image_classify.keras')

data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum',
 'carrot','cauliflower','chilli pepper','corn','cucumber','eggplant',
 'garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango',
 'onion','orange','paprika','pear','peas','pineapple','pomegranate',
 'potato','raddish','soy beans','spinach','sweetcorn','sweetpotato',
 'tomato','turnip','watermelon']

img_height = 180
img_width = 180

# 🔘 Input option
option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

image = None

# 📂 Upload
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# 📷 Camera
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

# 🚀 Prediction
if image is not None:
    st.image(image, caption="Input Image",use_column_width=True)

    # Preprocess
    image = image.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    # Predict
    prediction = model.predict(img_bat)
    score = tf.nn.softmax(prediction[0])

    # 🏆 Top prediction
    top_index = np.argmax(score)
    st.success(f"Prediction: {data_cat[top_index]}")
    st.info(f"Confidence: {score[top_index]*100:.2f}%")

    # 🔝 Top 3 predictions
    top_3_idx = np.argsort(score)[-3:][::-1]
    st.subheader("Top 3 Predictions")

    top_data = {
        "Category": [data_cat[i] for i in top_3_idx],
        "Confidence (%)": [float(score[i]*100) for i in top_3_idx]
    }

    df = pd.DataFrame(top_data)
    st.table(df)

  