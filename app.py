import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="SmartRoad AI - Model Comparison",
    page_icon="🛣️",
    layout="wide"
)

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Model not found at {path}! {e}")
        return None

def analyze_severity(area, image_area):
    relative_area = (area / image_area) * 100
    if relative_area < 0.5:
        return "🟢 Low", "#28a745"
    elif relative_area < 2.0:
        return "🟡 Medium", "#ffc107"
    else:
        return "🔴 High", "#dc3545"

def process_image(image, model, confidence):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = opencv_image.shape[:2]
    image_area = height * width

    if model is None:
        return image, []

    results = model(opencv_image, conf=confidence)
    detections = []
    annotated_image = opencv_image.copy()

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                area = (x2 - x1) * (y2 - y1)
                severity_text, severity_color = analyze_severity(area, image_area)
                if severity_text.startswith("🟢"):
                    cv_color = (0, 255, 0)
                elif severity_text.startswith("🟡"):
                    cv_color = (0, 255, 255)
                else:
                    cv_color = (0, 0, 255)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'area': area,
                    'severity': severity_text,
                    'confidence': float(box.conf[0]),
                    'relative_area': (area / image_area) * 100
                })
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), cv_color, 3)
                label = f"Pothole: {severity_text.split()[1]} ({box.conf[0]:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_image, (x1, y1-label_size[1]-10),
                              (x1+label_size[0], y1), cv_color, -1)
                cv2.putText(annotated_image, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image), detections

def main():
    st.markdown("<h1 class='main-header'>🛣️ SmartRoad AI: Multi-Model Pothole Detection</h1>", unsafe_allow_html=True)
    st.sidebar.header("⚙️ Configuration")
    confidence = st.sidebar.slider(
        "Detection Confidence", min_value=0.1, max_value=1.0,
        value=0.3, step=0.05,
        help="Lower values detect more potholes but may include false positives"
    )

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a road image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a road surface"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner('Model 1 (best.pt) running...'):
                model1 = load_model('best.pt')
                processed_img1, detections1 = process_image(image, model1, confidence)
            st.subheader("Model 1 - Results (best.pt)")
            st.image(processed_img1, caption="Model 1 Results", use_column_width=True)

    with col2:
        if uploaded_file is not None:
            with st.spinner('Model 2 (best-2.pt) running...'):
                model2 = load_model('best-2.pt')
                processed_img2, detections2 = process_image(image, model2, confidence)
            st.subheader("Model 2 - Results (best-2.pt)")
            st.image(processed_img2, caption="Model 2 Results", use_column_width=True)

    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("Detection Comparison Table")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Model 1 (best.pt):**")
            st.write(f"Total Detections: {len(detections1)}")
            st.write(f"Low: {sum(1 for d in detections1 if d['severity'].startswith('🟢'))}")
            st.write(f"Medium: {sum(1 for d in detections1 if d['severity'].startswith('🟡'))}")
            st.write(f"High: {sum(1 for d in detections1 if d['severity'].startswith('🔴'))}")
        with col_b:
            st.markdown("**Model 2 (best-2.pt):**")
            st.write(f"Total Detections: {len(detections2)}")
            st.write(f"Low: {sum(1 for d in detections2 if d['severity'].startswith('🟢'))}")
            st.write(f"Medium: {sum(1 for d in detections2 if d['severity'].startswith('🟡'))}")
            st.write(f"High: {sum(1 for d in detections2 if d['severity'].startswith('🔴'))}")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🎓 Final Year BCA Data Science Project | SmartRoad AI System</p>
        <p>Multi-model YOLOv8 comparison</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
