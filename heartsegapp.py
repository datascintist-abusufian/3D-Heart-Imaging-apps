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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json

# Constants and Configuration
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"
MODEL_PATH = "models/yolov8x-seg.pt"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Page Configuration
st.set_page_config(
    page_title="Heart MRI Analysis",
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
    .debug-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-family: monospace;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00a6ed;
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
    """Load YOLO segmentation model"""
    try:
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading segmentation model..."):
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")
        
        with st.spinner("🔄 Loading model..."):
            model = YOLO(MODEL_PATH)
            model.conf = CONFIDENCE_THRESHOLD
            model.task = 'segment'
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def process_image(image):
    """Process image for model input"""
    try:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(image)
        return img_array, image
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def process_segmentation(image, model):
    """Process segmentation and create visualizations"""
    try:
        # Run segmentation
        results = model(image, task='segment')
        
        if len(results) > 0:
            result = results[0]
            img_draw = np.array(image).copy()
            
            # Process masks and boxes
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                
                # Create segmentation visualization
                overlay = np.zeros_like(img_draw, dtype=np.uint8)
                stats = []
                
                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    # Resize mask to image size
                    mask = cv2.resize(mask.squeeze(), (image.shape[1], image.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    # Create colored overlay
                    color = np.array([0, 255, 0], dtype=np.uint8)  # Green
                    overlay[mask > 0] = color
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = float(box[4])
                    cls_id = int(box[5])
                    
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    cv2.putText(img_draw, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Calculate statistics
                    area = np.sum(mask)
                    stats.append({
                        'Class': CLASS_NAMES[cls_id],
                        'Confidence': conf,
                        'Area': area,
                        'Mask': mask
                    })
                
                # Combine image with overlay
                alpha = 0.5
                segmented = cv2.addWeighted(img_draw, 1, overlay, alpha, 0)
                
                return segmented, stats, masks
                
        return None, None, None
        
    except Exception as e:
        st.error(f"❌ Error in segmentation: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))
        return None, None, None

def create_3d_visualization(mask):
    """Create 3D visualization of segmentation mask"""
    try:
        if mask is None:
            return None
            
        # Create surface plot
        z_data = mask.squeeze()
        x, y = np.meshgrid(
            np.linspace(0, z_data.shape[1], z_data.shape[1]),
            np.linspace(0, z_data.shape[0], z_data.shape[0])
        )
        
        fig = go.Figure(data=[go.Surface(z=z_data, x=x, y=y)])
        fig.update_layout(
            title='3D Segmentation Visualization',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Mask Intensity'
            ),
            width=800,
            height=800
        )
        return fig
    except Exception as e:
        st.error(f"❌ Error creating 3D visualization: {str(e)}")
        return None

def create_metrics_dashboard(stats):
    """Create analysis dashboard"""
    try:
        if not stats:
            return
            
        # Calculate metrics
        metrics = {
            'Total Regions': len(stats),
            'Average Confidence': np.mean([s['Confidence'] for s in stats]),
            'Total Area': sum([s['Area'] for s in stats]),
            'Average Area': np.mean([s['Area'] for s in stats])
        }
        
        # Display metrics
        cols = st.columns(len(metrics))
        for (metric_name, value), col in zip(metrics.items(), cols):
            with col:
                st.metric(
                    label=metric_name,
                    value=f"{value:.2f}" if isinstance(value, float) else value
                )
        
        # Create detailed analysis
        if len(stats) > 0:
            df = pd.DataFrame([{
                'Class': s['Class'],
                'Confidence': s['Confidence'],
                'Area': s['Area']
            } for s in stats])
            
            st.markdown("### Detailed Analysis")
            st.dataframe(df)
            
            # Plot distributions
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(df, x='Confidence', title='Confidence Distribution')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.pie(df, names='Class', values='Area', title='Area Distribution by Class')
                st.plotly_chart(fig2, use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")

def process_and_visualize(image, model):
    """Main processing and visualization pipeline"""
    try:
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "🎯 Segmentation",
            "📊 Analysis",
            "📈 3D View"
        ])
        
        # Process image and run segmentation
        with st.spinner("Processing image..."):
            processed_img, original_img = process_image(image)
            if processed_img is None:
                return
                
            segmented_img, stats, masks = process_segmentation(processed_img, model)
            
            if segmented_img is not None:
                with tab1:
                    st.markdown("### Segmentation Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(original_img, caption="Original Image")
                    with col2:
                        st.image(segmented_img, caption="Segmentation Result")
                
                with tab2:
                    st.markdown("### Analysis Results")
                    create_metrics_dashboard(stats)
                
                with tab3:
                    st.markdown("### 3D Visualization")
                    if masks is not None and len(masks) > 0:
                        fig = create_3d_visualization(masks[0])
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No segmentation results found in the image")
                
    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))

def main():
    """Main application function"""
    st.title("🫀 Heart MRI Segmentation Analysis")
    st.markdown("""
        This application performs segmentation analysis on heart MRI images using 
        deep learning to identify and visualize cardiac structures.
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Image source selection
        src = st.radio(
            "📷 Select Image Source",
            ['Upload Image', 'Sample Images'],
            help="Choose image source"
        )
        
        # Model settings
        st.markdown("### ⚙️ Model Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05
        )
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode")
        
        st.markdown("---")
        st.markdown("""
            ### 📋 Information
            - Model: YOLOv8-Segmentation
            - Input Size: 640x640
            - Task: Cardiac Structure Segmentation
        """)
    
    # Load model
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("❌ Model loading failed. Please check your connection.")
        return
        
    # Update model settings
    st.session_state.model.conf = conf_threshold
    
    # Image processing
    try:
        if src == 'Upload Image':
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=300)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    process_and_visualize(image, st.session_state.model)
        else:
            # Sample images
            selected_image = st.slider(
                "Select sample image",
                1, 10,
                help="Choose from sample dataset"
            )
            
            sample_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{selected_image}.jpg"
            
            try:
                response = requests.get(sample_url)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content)).convert("RGB")
                st.image(image, caption=f"Sample Image #{selected_image}", width=300)
                
                if st.button("🔍 Analyze Sample", type="primary"):
                    process_and_visualize(image, st.session_state.model)
                    
            except Exception as e:
                st.error(f"❌ Error loading sample image: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ Error in application: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Created by Md Abu Sufian | Version 2.0</p>
            <p>For research and educational purposes only</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
