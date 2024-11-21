import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import os
import requests
from torchvision.transforms import transforms
from io import BytesIO
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

# Constants and Configuration
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
MODEL_PATH = "models/yolov5s.pt"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Page Configuration
st.set_page_config(
    page_title="3D Heart MRI Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #00a6ed;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
    }
    .analysis-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #31333F;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

@st.cache_resource
def load_model():
    """Load YOLO model and handle errors."""
    try:
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model..."):
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")
        with st.spinner("🔄 Loading model..."):
            model = YOLO(MODEL_PATH)
            model.conf = CONFIDENCE_THRESHOLD
            model.iou = 0.45
            model.max_det = 100
            return model
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error downloading model: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
    return None

def process_image(image):
    """Preprocess image for inference."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(image) / 255.0
        return img_array, image
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def draw_detections(image, results):
    """Draw bounding boxes and handle errors."""
    try:
        img_draw = np.array(image).copy()
        confidence_scores = []
        detection_stats = []
        pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        if hasattr(results, 'boxes') and results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf > CONFIDENCE_THRESHOLD:
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    confidence_scores.append(conf)
                    detection_stats.append({'Class': CLASS_NAMES[cls_id], 'Confidence': conf, 'Area': (x2 - x1) * (y2 - y1)})
        return img_draw, confidence_scores, pred_mask, detection_stats
    except Exception as e:
        st.error(f"❌ Error drawing detections: {str(e)}")
        return image, [], np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8), []

def process_and_visualize(image, model):
    """Run model inference and visualize results."""
    try:
        with st.spinner("🔄 Running model inference..."):
            results = model(image)
            if not results or not results[0].boxes:
                st.warning("No detections found.")
                return
            result = results[0]
            img_with_boxes, confidence_scores, pred_mask, detection_stats = draw_detections(image, result)
            
            st.markdown("### 🎯 Detection Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image")
            with col2:
                st.image(img_with_boxes, caption="Detections")
            
            if confidence_scores:
                st.markdown("### 📊 Confidence Distribution")
                fig = px.histogram(x=confidence_scores, nbins=10, title="Confidence Scores", labels={'x': 'Confidence'})
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        if st.session_state.debug_mode:
            st.write(f"Debug: {e}")

def main():
    st.title("🫀 3D Heart MRI Analysis")
    st.markdown("Upload an MRI image to analyze cardiac structures.")
    
    if 'model' not in st.session_state or not st.session_state.model:
        st.session_state.model = load_model()
    
    if st.session_state.model:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("🔍 Analyze Image"):
                process_and_visualize(image, st.session_state.model)
    else:
        st.error("Model loading failed. Check your connection and try again.")

if __name__ == "__main__":
    main()
