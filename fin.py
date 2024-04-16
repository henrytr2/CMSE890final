import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load your pre-trained model
# Assuming you've already loaded your model and named it loaded_model

# Function to preprocess and predict the uploaded image
def predict_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert image to BGR format
    img = cv2.resize(img, (256, 224))  # Resize image to match model input size
    img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values
    
    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_name = "glottis" if predicted_class == 0 else "no glottis"
    
    return class_name

# Streamlit app
st.title("Image Classification")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict_image(image)
    st.write(f"Prediction: {prediction}")
