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
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import time

# --- Constants and Configuration ---
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
MODEL_PATH = "yolov5s.pt"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
THRESHOLD = 0.3

st.set_page_config(
    page_title="3D Heart MRI Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model with proper error handling"""
    try:
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', MODEL_PATH)

        if not os.path.exists(model_path):
            with st.spinner("📥 Downloading model..."):
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")

        with st.spinner("🔄 Loading model..."):
            model = YOLO(model_path)
            st.success("✅ Model loaded successfully!")
            return model

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

def create_3d_surface_plot(mask):
    """Create 3D surface plot of segmentation mask"""
    x, y = np.meshgrid(
        np.linspace(0, mask.shape[1], mask.shape[1]),
        np.linspace(0, mask.shape[0], mask.shape[0])
    )
    
    fig = go.Figure(data=[go.Surface(z=mask, x=x, y=y)])
    fig.update_layout(
        title='3D Surface Plot of Segmentation Mask',
        scene=dict(
            xaxis_title='Width',
            yaxis_title='Height',
            zaxis_title='Intensity'
        ),
        width=600,
        height=500
    )
    return fig

def create_confidence_plot(confidence_scores):
    """Create confidence distribution plot"""
    if not confidence_scores:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=confidence_scores,
        nbinsx=20,
        name='Confidence Distribution',
        marker_color='#00a6ed'
    ))
    fig.update_layout(
        title='Detection Confidence Distribution',
        xaxis_title='Confidence Score',
        yaxis_title='Frequency',
        showlegend=False,
        width=600,
        height=400
    )
    return fig

def calculate_metrics(true_mask, pred_mask):
    """Calculate evaluation metrics"""
    try:
        true_mask = cv2.resize(true_mask, (640, 640))
        pred_mask_binary = (pred_mask > 0).astype(np.uint8)
        true_mask_binary = (true_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(true_mask_binary, pred_mask_binary)
        union = np.logical_or(true_mask_binary, pred_mask_binary)
        
        dice = 2 * np.sum(intersection) / (np.sum(pred_mask_binary) + np.sum(true_mask_binary) + 1e-6)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        
        precision = precision_score(true_mask_binary.flatten(), pred_mask_binary.flatten(), zero_division=0)
        recall = recall_score(true_mask_binary.flatten(), pred_mask_binary.flatten(), zero_division=0)
        f1 = f1_score(true_mask_binary.flatten(), pred_mask_binary.flatten(), zero_division=0)
        
        return dice, iou, precision, recall, f1
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        return 0, 0, 0, 0, 0

def create_metrics_dashboard(metrics):
    """Create metrics dashboard with styled cards"""
    cols = st.columns(5)
    
    metric_styles = {
        'Dice Score': ('🎯', '#FF6B6B'),
        'IoU': ('🔄', '#4ECDC4'),
        'Precision': ('📊', '#45B7D1'),
        'Recall': ('📈', '#96CEB4'),
        'F1 Score': ('⭐', '#FFEEAD')
    }
    
    for (metric_name, value), col in zip(metrics.items(), cols):
        icon, color = metric_styles[metric_name]
        with col:
            st.markdown(f"""
            <div style='
                background-color: {color}22;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border: 2px solid {color};
            '>
                <h3 style='margin: 0; color: #31333F;'>{icon} {metric_name}</h3>
                <p style='font-size: 24px; margin: 10px 0; color: {color};'>{value:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

def process_and_visualize(image, model):
    """Process image and create visualizations"""
    with st.spinner("🔄 Processing image..."):
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🎯 Segmentation", "📈 3D View"])
        
        # Process image
        img_tensor = process_image(image)
        if img_tensor is None:
            return
        
        # Run inference
        try:
            results = model(img_tensor)[0]
            img_with_bboxes, confidence_scores, pred_mask = draw_bboxes_and_masks(image, results)
            
            with tab1:
                st.subheader("Analysis Results")
                if np.any(pred_mask):
                    ground_truth_mask = np.zeros((640, 640), dtype=np.uint8)
                    dice, iou, precision, recall, f1 = calculate_metrics(ground_truth_mask, pred_mask)
                    
                    metrics = {
                        'Dice Score': dice,
                        'IoU': iou,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    }
                    create_metrics_dashboard(metrics)
                    
                    conf_fig = create_confidence_plot(confidence_scores)
                    if conf_fig:
                        st.plotly_chart(conf_fig, use_container_width=True)
                else:
                    st.warning("⚠️ No detections found in the image")
            
            with tab2:
                st.subheader("Segmentation Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_with_bboxes, caption='Detected Regions', use_column_width=True)
                with col2:
                    mask_overlay = np.zeros_like(img_with_bboxes)
                    mask_overlay[pred_mask == 255] = (0, 255, 0)
                    combined_img = cv2.addWeighted(img_with_bboxes, 0.7, mask_overlay, 0.3, 0)
                    st.image(combined_img, caption='Segmentation Overlay', use_column_width=True)
            
            with tab3:
                st.subheader("3D Visualization")
                if np.any(pred_mask):
                    surface_plot = create_3d_surface_plot(pred_mask)
                    st.plotly_chart(surface_plot, use_container_width=True)
                else:
                    st.warning("⚠️ No mask data available for 3D visualization")
                    
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")

def main():
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.markdown("---")
        
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
    
    model = load_model()
    if model is None:
        st.warning("⚠️ Please ensure you have a stable internet connection and try reloading the page.")
        return

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
