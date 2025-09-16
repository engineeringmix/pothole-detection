import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Set page configuration
st.set_page_config(page_title="Pothole Detector", layout="wide")

# Title and description
st.title("Pothole Detection with YOLOv8")
st.markdown("Upload an image to detect potholes using a pre-trained YOLOv8 model. Results will display bounding boxes around detected potholes with confidence scores.")

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Model file 'best.pt' not found. Please ensure it is in the same directory as this script.")
        return None
    return YOLO(model_path)

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Read and process the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convert to RGB if necessary (YOLO expects RGB)
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Run inference
    with st.spinner("Detecting potholes..."):
        results = model.predict(image_np, conf=0.5)  # Adjust confidence threshold as needed
    
    # Process results
    result_img = results[0].plot()  # YOLOv8's built-in plotting for bounding boxes and labels
    
    # Display original and result images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(result_img, caption="Detection Results", use_column_width=True)
    
    # Display detection details
    st.subheader("Detection Details")
    detections = results[0].boxes
    if len(detections) > 0:
        for i, box in enumerate(detections):
            class_name = results[0].names[int(box.cls)]
            confidence = box.conf.item()
            st.write(f"Pothole {i+1}: Confidence = {confidence:.2f}")
    else:
        st.write("No potholes detected.")

# Instructions if no file is uploaded
if uploaded_file is None:
    st.info("Please upload an image to start the detection.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and YOLOv8 | Powered by xAI")
