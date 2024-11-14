import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
from torchvision.transforms import transforms
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import jaccard_score, precision_score, recall_score
import time

# --- Constants ---
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
MODEL_PATH = "yolov5s.pt"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
THRESHOLD = 0.3

# --- Page Configuration ---
st.set_page_config(
    page_title="3D Heart MRI Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model with proper error handling"""
    try:
        # Create a models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', MODEL_PATH)

        # Download the model if it doesn't exist
        if not os.path.exists(model_path):
            with st.spinner("📥 Downloading model..."):
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()  # Raise an error for bad responses
                
                # Save the model file
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")

        # Load the model
        with st.spinner("🔄 Loading model..."):
            model = YOLO(model_path)
            st.success("✅ Model loaded successfully!")
            return model

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def process_image(image):
    """Process the input image"""
    try:
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None

def draw_bboxes_and_masks(image, results):
    """Draw bounding boxes and masks on the image"""
    img = np.array(image)
    confidence_scores = []
    pred_mask = np.zeros((640, 640), dtype=np.uint8)

    try:
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, 'Unknown')
                
                confidence_scores.append(conf)
                
                if conf > THRESHOLD:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    if results.masks is not None and i < len(results.masks):
                        mask = results.masks[i].data.cpu().numpy()[0]
                        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                        mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                        pred_mask[y1:y2, x1:x2] = mask_resized

        return img, confidence_scores, pred_mask
    except Exception as e:
        st.error(f"❌ Error processing detection results: {str(e)}")
        return img, [], pred_mask

def main():
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.markdown("---")
        
        # Image source selection
        src = st.radio(
            "📷 Select Image Source",
            ['Sample Gallery', 'Upload Image'],
            help="Choose whether to use a sample image or upload your own"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📋 Information
        - Model: YOLOv5
        - Classes: Left & Right Ventricle
        - Resolution: 640x640
        """)

    # Main content
    st.title("🫀 3D MRI Heart Analysis Dashboard")
    st.markdown("""
    This advanced dashboard provides comprehensive analysis of heart MRI images using 
    deep learning for segmentation and detection of cardiac structures.
    """)
    
    # Load model with proper error handling
    model = load_model()
    if model is None:
        st.warning("⚠️ Please make sure you have a stable internet connection and try reloading the page.")
        return

    # Image processing
    if src == 'Upload Image':
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a heart MRI image for analysis"
        )
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image', width=300)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    process_and_visualize(image, model)
            except Exception as e:
                st.error(f"❌ Error loading uploaded image: {str(e)}")
    
    else:
        try:
            selected_image = st.slider(
                "Select sample image",
                1, 50,
                help="Choose a sample image from our test dataset"
            )
            
            image_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{selected_image}.jpg"
            response = requests.get(image_url)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption=f'Sample Image #{selected_image}', width=300)
            
            if st.button("🔍 Analyze Image", type="primary"):
                process_and_visualize(image, model)
                
        except Exception as e:
            st.error(f"❌ Error loading sample image: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Created by Md Abu Sufian | Version 2.0</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
