
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
# Load your model
@st.cache_resource
def load_model():
    return YOLO('best.pt')  # Update with path to your trained model
model = load_model()
def get_severity(area, img_area):
    rel_area = (area / img_area) * 100
    if rel_area < 0.5:
        return "Low", (0, 255, 0)
    elif rel_area < 2.0:
        return "Medium", (0, 255, 255)
    else:
        return "High", (0, 0, 255)
st.title("SmartRoad AI: Pothole Detection and Severity Analysis")
uploaded_file = st.file_uploader("Upload a road image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_cv2.shape[:2]
    img_area = h * w
    results = model(img_cv2)
    boxes = results[0].boxes
    counts = {'Low': 0, 'Medium': 0, 'High': 0}
    # Draw results
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        severity, color = get_severity(area, img_area)
        counts[severity] += 1
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        text = f"{severity} ({box.conf[0]:.2f})"
        cv2.putText(img_cv2, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img_output = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    st.image(img_output, caption="Detection Result", use_column_width=True)
    st.write(f"### Results\n- Low Severity: {counts['Low']}\n- Medium Severity: {counts['Medium']}\n- High Severity: {counts['High']}")
else:
    st.info("Please upload an image to start")
st.markdown("---")
st.caption("Final Year BCA Data Science Project | SmartRoad AI System")
