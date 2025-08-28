import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configure page
st.set_page_config(
    page_title="SmartRoad AI - Pothole Detection",
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
</style>
""", unsafe_allow_html=True)


# Load YOLOv8 PyTorch model
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")  # your trained model
    except Exception as e:
        st.error("Model not found or failed to load. Check the .pt file path.")
        st.error(str(e))
        return None


def analyze_severity(area, image_area):
    """Classify pothole severity by relative area size"""
    relative_area = (area / image_area) * 100
    if relative_area < 0.5:
        return "🟢 Low"
    elif relative_area < 2.0:
        return "🟡 Medium"
    else:
        return "🔴 High"


def process_image(image, model, confidence):
    """Run YOLOv8 inference on an image"""
    np_img = np.array(image)  # PIL → NumPy
    h, w = np_img.shape[:2]
    image_area = h * w

    results = model.predict(source=np_img, conf=confidence, save=False, verbose=False)
    result = results[0]

    annotated = result.plot()  # YOLO gives ready-made annotated image (BGR)

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        area = (x2 - x1) * (y2 - y1)
        severity = analyze_severity(area, image_area)

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "severity": severity,
            "confidence": conf,
            "class": model.names[cls] if model.names else "pothole",
            "relative_area": (area / image_area) * 100
        })

    # Convert annotated image to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), detections


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
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Severity Levels
    - 🟢 **Low**: < 0.5% of image area
    - 🟡 **Medium**: 0.5% - 2% of image area  
    - 🔴 **High**: > 2% of image area
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose a road image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("🔍 Analyzing image for potholes..."):
                processed_image, detections = process_image(image, model, confidence)

            if processed_image:
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
                        st.write(f"**Location:** {detection['bbox']}")

            # Priority messages
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
        <p>Built with YOLOv8 (.pt), OpenCV, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
