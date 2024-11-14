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
import seaborn as sns
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
import time

# --- Configuration ---
st.set_page_config(page_title="3D Heart MRI Analysis", layout="wide", initial_sidebar_state="expanded")

# Constants
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
MODEL_PATH = "yolov5s.pt"
GIF_PATH = "WholeHeartSegment_ErrorMap_WhiteBg.gif"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
THRESHOLD = 0.3

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #00a6ed;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        model_file = MODEL_PATH
        if not os.path.exists(model_file):
            with st.status("Downloading model...") as status:
                response = requests.get(MODEL_URL)
                with open(model_file, 'wb') as f:
                    f.write(response.content)
                status.update(label="Model downloaded successfully!", state="complete")
        
        try:
            model = YOLO(model_file)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    try:
        return transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def create_analysis_metrics(dice, iou, precision, recall, f1):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Dice Score", f"{dice:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("IoU Score", f"{iou:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision", f"{precision:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", f"{recall:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1 Score", f"{f1:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

def plot_confidence_distribution(confidence_scores):
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
        template='plotly_white',
        showlegend=False,
        height=300
    )
    return fig

def create_3d_visualization(pred_mask):
    x, y = np.meshgrid(np.linspace(0, pred_mask.shape[1], pred_mask.shape[1]),
                      np.linspace(0, pred_mask.shape[0], pred_mask.shape[0]))
    
    fig = go.Figure(data=[go.Surface(z=pred_mask, x=x, y=y)])
    fig.update_layout(
        title='3D Visualization of Segmentation Mask',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Intensity'
        ),
        height=500
    )
    return fig

def process_and_analyze(image, model):
    with st.spinner("Processing image..."):
        img_tensor = process_image(image)
        if img_tensor is None:
            return
        
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        results = model(img_tensor)[0]
        img_with_bboxes, confidence_scores, pred_mask = draw_bboxes_and_masks(image, results)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🎯 Detections", "📈 3D View"])
        
        with tab1:
            if np.any(pred_mask):
                ground_truth_mask = np.zeros((640, 640), dtype=np.uint8)
                dice, iou, precision, recall, f1 = calculate_metrics(ground_truth_mask, pred_mask)
                create_analysis_metrics(dice, iou, precision, recall, f1)
                
                conf_fig = plot_confidence_distribution(confidence_scores)
                st.plotly_chart(conf_fig, use_container_width=True)
            else:
                st.warning("No masks detected in the image.")
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_with_bboxes, caption='Detection Results', use_column_width=True)
            with col2:
                mask_overlay = np.zeros_like(img_with_bboxes)
                mask_overlay[pred_mask == 255] = (0, 255, 0)
                combined_img = cv2.addWeighted(img_with_bboxes, 0.7, mask_overlay, 0.3, 0)
                st.image(combined_img, caption='Segmentation Overlay', use_column_width=True)
        
        with tab3:
            visualization_3d = create_3d_visualization(pred_mask)
            st.plotly_chart(visualization_3d, use_container_width=True)

def main():
    # Sidebar
    st.sidebar.title('🎛️ Control Panel')
    st.sidebar.markdown('---')
    
    # Main content
    st.title("🫀 3D MRI Heart Analysis Dashboard")
    st.markdown("""
    This advanced dashboard provides comprehensive analysis of heart MRI images using 
    deep learning for segmentation and detection of cardiac structures.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please try again.")
        return
    
    # Image source selection
    src = st.sidebar.radio(
        "📷 Select Image Source",
        ['Upload Image', 'Sample Gallery'],
        help="Choose whether to upload your own image or use one from our sample gallery"
    )
    
    if src == 'Upload Image':
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a heart MRI image for analysis"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
            
            if st.button("🔍 Analyze Image"):
                process_and_analyze(image, model)
    
    else:
        st.sidebar.markdown("### Sample Image Selection")
        selected_image = st.sidebar.slider(
            "Select image number",
            1, 50,
            help="Choose a sample image from our test dataset"
        )
        
        try:
            image_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{selected_image}.jpg"
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption=f'Sample Image #{selected_image}', use_column_width=False, width=300)
            
            if st.button("🔍 Analyze Image"):
                process_and_analyze(image, model)
                
        except Exception as e:
            st.error(f"Error loading sample image: {e}")

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown("""
    💡 **About**
    - Built with Streamlit & YOLOv5
    - Made by Md Abu Sufian
    - For research purposes only
    """)

if __name__ == '__main__':
    main()
