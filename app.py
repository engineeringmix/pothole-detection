import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Configure page
st.set_page_config(
    page_title="AI Pothole Detection & Severity Analysis",
    page_icon="🛣️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    margin-bottom: 2rem;
}
.detection-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')  # Update with your trained model path if needed
    except Exception as e:
        st.error("Model not found! Please upload your trained model file.")
        st.error(e)
        return None

def analyze_severity(area, image_area):
    """Analyze pothole severity"""
    relative_area = (area / image_area) * 100
    if relative_area < 0.5:
        return "🟢 Low", "#28a745"
    elif relative_area < 2.0:
        return "🟡 Medium", "#ffc107"
    else:
        return "🔴 High", "#dc3545"

def process_image(image, model, confidence):
    """Process image and detect potholes"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = opencv_image.shape[:2]
    image_area = height * width

    results = model(opencv_image, conf=confidence)
    detections = []
    annotated_image = opencv_image.copy()

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                area = (x2 - x1) * (y2 - y1)
                severity_text, severity_color = analyze_severity(area, image_area)
                # OpenCV color
                if severity_text.startswith("🟢"):
                    cv_color = (0, 255, 0)
                elif severity_text.startswith("🟡"):
                    cv_color = (0, 255, 255)
                else:
                    cv_color = (0, 0, 255)
                # Detections
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'area': area,
                    'severity': severity_text,
                    'confidence': float(box.conf[0]),
                    'relative_area': (area / image_area) * 100
                })
                # Draw
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
    st.markdown("<h1 class='main-header'>🛣️ SmartRoad AI: Pothole Detection System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 18px; color: #666;'>
        Upload a road image to detect potholes and analyze their severity using AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower values detect more potholes but may include false positives"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Severity Levels
    - 🟢 **Low**: < 0.5% of image area
    - 🟡 **Medium**: 0.5% - 2% of image area  
    - 🔴 **High**: > 2% of image area
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a road image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a road surface"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner('🔍 Analyzing image for potholes...'):
                processed_image, detections = process_image(image, model, confidence)

            st.subheader("🎯 Detection Results")
            st.image(processed_image, caption="Processed Image with Detections", use_column_width=True)

    with col2:
        if 'detections' in locals() and uploaded_file is not None:
            st.subheader("📈 Analysis Summary")

            total_potholes = len(detections)
            low_count = sum(1 for d in detections if d['severity'].startswith("🟢"))
            medium_count = sum(1 for d in detections if d['severity'].startswith("🟡"))
            high_count = sum(1 for d in detections if d['severity'].startswith("🔴"))

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Potholes", total_potholes)
                st.metric("🟢 Low Severity", low_count)
            with col_b:
                st.metric("🟡 Medium Severity", medium_count)
                st.metric("🔴 High Severity", high_count)

            if detections:
                st.subheader("🔍 Detailed Detection Info")
                for i, detection in enumerate(detections, 1):
                    with st.expander(f"Pothole #{i} - {detection['severity'].split()[1]} Severity"):
                        st.write(f"**Confidence:** {detection['confidence']:.2%}")
                        st.write(f"**Area:** {detection['area']:,} pixels")
                        st.write(f"**Relative Size:** {detection['relative_area']:.2f}% of image")
                        st.write(f"**Location:** ({detection['bbox'][0]}, {detection['bbox'][1]}) to "
                                 f"({detection['bbox'][2]}, {detection['bbox'][3]})")

            if high_count > 0:
                st.error(f"⚠️ **HIGH PRIORITY**: {high_count} severe pothole(s) detected!")
                st.write("Immediate maintenance recommended.")
            elif medium_count > 0:
                st.warning(f"⚠️ **MEDIUM PRIORITY**: {medium_count} moderate pothole(s) detected.")
                st.write("Schedule maintenance within 2-4 weeks.")
            elif low_count > 0:
                st.info(f"ℹ️ **LOW PRIORITY**: {low_count} minor pothole(s) detected.")
                st.write("Monitor and schedule routine maintenance.")
            else:
                st.success("✅ **EXCELLENT**: No significant potholes detected!")
        else:
            st.info("👆 Upload an image to see analysis results")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🎓 Final Year BCA Data Science Project | SmartRoad AI System</p>
        <p>Built with YOLOv8, OpenCV, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
