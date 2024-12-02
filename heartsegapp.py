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
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Verify image dimensions and channels
        if len(img_array.shape) != 3:
            raise ValueError("Image must have 3 channels")
        
        if img_array.shape[2] != 3:
            raise ValueError("Image must be RGB")
            
        # Resize image
        resized_image = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Verify the resize was successful
        if resized_image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Image resizing failed")
            
        return resized_image, image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))
            st.write("Original image shape:", getattr(image, 'size', None))
        return None, None

def process_segmentation(image, model):
    """Process segmentation and create visualizations"""
    try:
        # Run segmentation
        results = model(image, task='segment')
        
        # Check if results exist and have valid predictions
        if not results or len(results) == 0:
            st.warning("No predictions found in the image")
            return None, None, None
            
        result = results[0]
        
        # Verify masks exist
        if not hasattr(result, 'masks') or result.masks is None or len(result.masks) == 0:
            st.warning("No segmentation masks detected in the image")
            return None, None, None
            
        img_draw = np.array(image).copy()
        
        # Process masks and boxes
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        
        if len(masks) == 0 or len(boxes) == 0:
            st.warning("No valid segmentation results found")
            return None, None, None
            
        # Create segmentation visualization
        overlay = np.zeros_like(img_draw, dtype=np.uint8)
        stats = []
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            try:
                # Verify mask dimensions
                if mask.size == 0:
                    continue
                    
                # Resize mask to image size
                mask = cv2.resize(mask.squeeze(), (image.shape[1], image.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
                
                # Verify box coordinates
                if not all(np.isfinite(box)):
                    continue
                    
                # Create colored overlay
                color = np.array([0, 255, 0], dtype=np.uint8)  # Green
                overlay[mask > 0] = color
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4])
                cls_id = int(box[5])
                
                # Verify class ID is valid
                if cls_id not in CLASS_NAMES:
                    continue
                    
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
                
            except Exception as e:
                if st.session_state.debug_mode:
                    st.warning(f"Error processing individual mask: {str(e)}")
                continue
        
        if not stats:
            st.warning("No valid segments could be processed")
            return None, None, None
            
        # Combine image with overlay
        alpha = 0.5
        segmented = cv2.addWeighted(img_draw, 1, overlay, alpha, 0)
        
        return segmented, stats, masks
        
    except Exception as e:
        st.error(f"❌ Error in segmentation: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))
            st.write("Results shape:", getattr(results[0] if results else None, 'shape', None))
            st.write("Image shape:", getattr(image, 'shape', None))
        return None, None, None

def create_metrics_dashboard(stats):
    """Create enhanced analysis dashboard with detailed cardiac metrics"""
    try:
        if not stats:
            st.warning("No statistics available for analysis")
            return
            
        # Calculate enhanced metrics
        ventricle_metrics = {
            class_name: {
                'area': sum(s['Area'] for s in stats if s['Class'] == class_name),
                'avg_confidence': np.mean([s['Confidence'] for s in stats if s['Class'] == class_name]),
                'count': len([s for s in stats if s['Class'] == class_name])
            }
            for class_name in CLASS_NAMES.values()
        }
        
        # Display summary metrics
        st.markdown("### 📊 Summary Metrics")
        cols = st.columns(2)
        
        for idx, (ventricle, metrics) in enumerate(ventricle_metrics.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{ventricle}</h3>
                    <p>Area: {metrics['area']:,.0f} pixels</p>
                    <p>Confidence: {metrics['avg_confidence']:.2%}</p>
                    <p>Regions Detected: {metrics['count']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Calculate ventricle ratio
        if all(metrics['area'] > 0 for metrics in ventricle_metrics.values()):
            lv_area = ventricle_metrics['Left Ventricle']['area']
            rv_area = ventricle_metrics['Right Ventricle']['area']
            ratio = lv_area / rv_area if rv_area > 0 else 0
            
            st.markdown("### 📏 Ventricle Ratio Analysis")
            st.markdown(f"""
            <div class="metric-card">
                <p>LV/RV Ratio: {ratio:.2f}</p>
                <p>This ratio can indicate potential cardiac conditions if significantly different from normal range (0.8-1.2)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create ratio gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ratio,
                gauge = {
                    'axis': {'range': [0, 2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.8], 'color': "lightgray"},
                        {'range': [0.8, 1.2], 'color': "lightgreen"},
                        {'range': [1.2, 2], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': ratio
                    }
                },
                title = {'text': "LV/RV Ratio"}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis if multiple frames
        if len(stats) > 1:
            st.markdown("### 📈 Temporal Analysis")
            
            # Create time series data
            time_data = pd.DataFrame([{
                'Frame': i,
                'Class': s['Class'],
                'Area': s['Area'],
                'Confidence': s['Confidence']
            } for i, s in enumerate(stats)])
            
            # Plot area changes
            fig_area = px.line(time_data, x='Frame', y='Area', color='Class',
                             title='Ventricle Area Over Time')
            st.plotly_chart(fig_area, use_container_width=True)
            
            # Plot confidence changes
            fig_conf = px.line(time_data, x='Frame', y='Confidence', color='Class',
                             title='Detection Confidence Over Time')
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Statistical analysis
        st.markdown("### 📊 Statistical Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create box plot of areas
            area_data = pd.DataFrame([{
                'Class': s['Class'],
                'Area': s['Area']
            } for s in stats])
            fig_box = px.box(area_data, x='Class', y='Area',
                           title='Area Distribution by Ventricle')
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Create scatter plot of confidence vs area
            fig_scatter = px.scatter(
                pd.DataFrame(stats),
                x='Confidence',
                y='Area',
                color='Class',
                title='Area vs Confidence',
                labels={'Area': 'Area (pixels)', 'Confidence': 'Detection Confidence'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Export results
        if st.button("📄 Export Analysis Report"):
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': ventricle_metrics,
                'ratio_analysis': {
                    'lv_rv_ratio': ratio,
                    'normal_range': {'min': 0.8, 'max': 1.2}
                },
                'raw_stats': stats
            }
            
            st.download_button(
                label="Download JSON Report",
                data=json.dumps(report_data, indent=2),
                file_name="heart_analysis_report.json",
                mime="application/json"
            )
            
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))

def create_3d_visualization(masks):
    """Create enhanced 3D visualization of segmentation masks"""
    try:
        if masks is None or len(masks) == 0:
            st.warning("No masks available for 3D visualization")
            return None
            
        # Create tabs for different 3D views
        view_tab1, view_tab2 = st.tabs(["Surface Plot", "Volume Rendering"])
        
        with view_tab1:
            # Surface plot
            for idx, mask in enumerate(masks):
                z_data = mask.squeeze()
                x, y = np.meshgrid(
                    np.linspace(0, z_data.shape[1], z_data.shape[1]),
                    np.linspace(0, z_data.shape[0], z_data.shape[0])
                )
                
                fig = go.Figure(data=[go.Surface(
                    z=z_data,
                    x=x,
                    y=y,
                    colorscale='Viridis',
                    name=f'Mask {idx+1}'
                )])
                
                fig.update_layout(
                    title=f'3D Surface Plot - {CLASS_NAMES.get(idx, f"Mask {idx+1}")}',
                    scene=dict(
                        xaxis_title='Width',
                        yaxis_title='Height',
                        zaxis_title='Intensity',
                        camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    width=800,
                    height=800
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with view_tab2:
            # Volume rendering view
            combined_mask = np.zeros_like(masks[0].squeeze())
            for idx, mask in enumerate(masks):
                combined_mask += (idx + 1) * mask.squeeze()
            
            fig = go.Figure(data=go.Volume(
                x=np.arange(combined_mask.shape[0]),
                y=np.arange(combined_mask.shape[1]),
                z=np.arange(1),
                value=combined_mask,
                opacity=0.5,
                surface_count=20,
                colorscale='Viridis'
            ))
            
            fig.update_layout(
                title='3D Volume Rendering',
                scene=dict(
                    xaxis_title='Width',
                    yaxis_title='Height',
                    zaxis_title='Depth'
                ),
                width=800,
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Add visualization controls
        st.sidebar.markdown("### 🎮 3D Controls")
        opacity = st.sidebar.slider("Opacity", 0.0, 1.0, 0.5, 0.1)
        colormap = st.sidebar.selectbox(
            "Color Scheme",
            ['Viridis', 'Plasma', 'Inferno', 'Magma']
        )
        
    except Exception as e:
        st.error(f"❌ Error creating 3D visualization: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))
        return None

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
                        create_3d_visualization(masks)
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
            - Model: AI model-Segmentation
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
            <p>Created with ❤️ for Medical Imaging Analysis</p>
            <p>For research and educational purposes only</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
