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
import seaborn as sns
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import time
import pandas as pd
from datetime import datetime

# --- Constants and Configuration ---
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
MODEL_PATH = "yolov5s.pt"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
CONFIDENCE_THRESHOLD = 0.25  # Lowered threshold for testing

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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model with debug information"""
    try:
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', MODEL_PATH)

        if not os.path.exists(model_path):
            with st.spinner("📥 Downloading model..."):
                st.write("Debug: Downloading model from URL")
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")

       with st.spinner("🔄 Loading model..."):
            st.write("Debug: Initializing YOLO model")
            model = YOLO(model_path)
            # Set model parameters
            model.conf = CONFIDENCE_THRESHOLD
            model.iou = 0.45  # Added IOU threshold
            model.max_det = 100  # Maximum detections
            model.verbose = False
            st.success("✅ Model loaded successfully!")
            return model

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.write("Debug: Full error details:", str(e))
        return None

def process_image(image):
    """Process the input image with enhanced preprocessing"""
    try:
        # Debug information
        st.write("Debug: Processing image")
        st.write(f"Debug: Input image size: {image.size}")
        st.write(f"Debug: Input image mode: {image.mode}")

        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            st.write("Debug: Converted image to RGB")

        # Create transform pipeline
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transformations
        img_tensor = transform(image).unsqueeze(0)
        st.write(f"Debug: Output tensor shape: {img_tensor.shape}")
        return img_tensor

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.write("Debug: Full error details:", str(e))
        return None

def draw_bboxes_and_masks(image, results):
    """Draw bounding boxes and masks with enhanced debugging"""
    try:
        img = np.array(image.copy())
        confidence_scores = []
        pred_mask = np.zeros((640, 640), dtype=np.uint8)
        detection_stats = []

        # Debug information
        st.write("Debug: Processing detections")
        st.write(f"Debug: Number of detections: {len(results.boxes)}")
        
        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            st.write(f"Debug: Detection scores: {[float(box.conf[0]) for box in results.boxes]}")
            
            for i, box in enumerate(results.boxes):
                # Get box coordinates and ensure they're within image bounds
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf > CONFIDENCE_THRESHOLD:
                    # Add to confidence scores
                    confidence_scores.append(conf)
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Handle masks if available
                    if hasattr(results, 'masks') and results.masks is not None:
                        try:
                            masks = results.masks.data.cpu().numpy()
                            if i < len(masks):
                                mask = masks[i]
                                mask = cv2.resize(mask, (x2 - x1, y2 - y1))
                                mask = (mask > 0.5).astype(np.uint8) * 255
                                pred_mask[y1:y2, x1:x2] = mask
                                
                                # Create mask overlay
                                roi = img[y1:y2, x1:x2]
                                roi[mask > 127] = roi[mask > 127] * 0.5 + np.array([0, 255, 0]) * 0.5
                                img[y1:y2, x1:x2] = roi
                                
                                st.write(f"Debug: Processed mask for detection {i}")
                        except Exception as e:
                            st.warning(f"Debug: Mask processing error for detection {i}: {str(e)}")
                    
                    # Add detection stats
                    detection_stats.append({
                        'Class': CLASS_NAMES[cls_id],
                        'Confidence': conf,
                        'Box Area': (x2-x1)*(y2-y1),
                        'Center X': (x1+x2)/2,
                        'Center Y': (y1+y2)/2
                    })

        if not confidence_scores:
            st.warning("Debug: No detections above confidence threshold")
            
        st.write(f"Debug: Processed {len(confidence_scores)} valid detections")
        return img, confidence_scores, pred_mask, detection_stats

    except Exception as e:
        st.error(f"❌ Error in detection processing: {str(e)}")
        st.write("Debug: Full error details:", str(e))
        return image, [], np.zeros((640, 640), dtype=np.uint8), []

def create_3d_surface_plot(mask):
    """Create 3D surface plot of segmentation mask"""
    try:
        x, y = np.meshgrid(
            np.linspace(0, mask.shape[1], mask.shape[1]),
            np.linspace(0, mask.shape[0], mask.shape[0])
        )
        
        fig = go.Figure(data=[go.Surface(z=mask, x=x, y=y)])
        fig.update_layout(
            title='3D Visualization of Segmentation Mask',
            scene=dict(
                xaxis_title='Width',
                yaxis_title='Height',
                zaxis_title='Intensity'
            ),
            width=None,
            height=600,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        return fig
    except Exception as e:
        st.error(f"❌ Error creating 3D plot: {str(e)}")
        return None

def create_confidence_plot(confidence_scores):
    """Create confidence distribution plot"""
    if not confidence_scores:
        return None
    
    try:
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
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"❌ Error creating confidence plot: {str(e)}")
        return None

def calculate_metrics(mask):
    """Calculate various metrics for the segmentation mask"""
    try:
        if not np.any(mask):
            return {}
        
        # Calculate basic properties
        total_pixels = mask.size
        mask_pixels = np.sum(mask > 0)
        coverage = mask_pixels / total_pixels
        
        # Calculate contour properties
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        metrics = {
            'Coverage (%)': coverage * 100,
            'Number of Regions': len(contours),
            'Total Area (pixels)': mask_pixels
        }
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            metrics.update({
                'Perimeter (pixels)': cv2.arcLength(largest_contour, True),
                'Compactness': 4 * np.pi * cv2.contourArea(largest_contour) / 
                              (cv2.arcLength(largest_contour, True) ** 2) 
                              if cv2.arcLength(largest_contour, True) > 0 else 0
            })
        
        return metrics
        
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        return {}

def create_metrics_dashboard(metrics):
    """Create metrics dashboard with styled cards"""
    try:
        if not metrics:
            return
        
        cols = st.columns(len(metrics))
        
        metric_styles = {
            'Coverage (%)': ('📊', '#FF6B6B'),
            'Number of Regions': ('🔢', '#4ECDC4'),
            'Total Area (pixels)': ('📏', '#45B7D1'),
            'Perimeter (pixels)': ('⭕', '#96CEB4'),
            'Compactness': ('🔄', '#FFEEAD')
        }
        
        for (metric_name, value), col in zip(metrics.items(), cols):
            icon, color = metric_styles.get(metric_name, ('📊', '#666666'))
            with col:
                st.markdown(f"""
                <div style='
                    background-color: {color}22;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    border: 2px solid {color};
                '>
                    <h3 style='margin: 0; color: #31333F; font-size: 1rem;'>{icon} {metric_name}</h3>
                    <p style='font-size: 1.5rem; margin: 10px 0; color: {color};'>
                        {value:.2f if isinstance(value, float) else value}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"❌ Error creating metrics dashboard: {str(e)}")

def process_and_visualize(image, model):
    """Process image and create visualizations"""
    try:
        with st.spinner("🔄 Processing image..."):
            # Add debug information
            st.write("Debug: Image size:", image.size)
            st.write("Debug: Model confidence threshold:", model.conf)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "📊 Analysis Results", 
                "🎯 Segmentation View", 
                "📈 3D Visualization"
            ])
            
            # Process image
            img_tensor = process_image(image)
            if img_tensor is None:
                return
            
            # Run inference
            st.write("Debug: Running model inference")
            results = model(img_tensor)[0]
            st.write("Debug: Number of detections:", len(results[0].boxes))
            
            # Process detections
            img_with_bboxes, confidence_scores, pred_mask, detection_stats = draw_bboxes_and_masks(image, results[0])
            st.write("Debug: Confidence scores:", confidence_scores)
            
            if len(confidence_scores) > 0:
                with tab1:
                    st.markdown("### 📊 Analysis Results")
                    
                    # Show metrics
                    metrics = calculate_metrics(pred_mask)
                    if metrics:
                        create_metrics_dashboard(metrics)
                    
                    # Show detection statistics
                    if detection_stats:
                        st.markdown("### 📋 Detection Details")
                        df = pd.DataFrame(detection_stats)
                        st.dataframe(df)
                        
                        # Show confidence distribution
                        st.markdown("### 📈 Confidence Distribution")
                        conf_fig = create_confidence_plot(confidence_scores)
                        if conf_fig:
                            st.plotly_chart(conf_fig, use_container_width=True)
                
                with tab2:
                    st.markdown("### 🎯 Segmentation Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Original Detection")
                        st.image(img_with_bboxes, caption='Detected Regions', use_column_width=True)
                    
                    with col2:
                        st.markdown("#### Segmentation Overlay")
                        if np.any(pred_mask):
                            mask_overlay = np.zeros_like(img_with_bboxes)
                            mask_overlay[pred_mask == 255] = (0, 255, 0)
                            combined_img = cv2.addWeighted(img_with_bboxes, 0.7, mask_overlay, 0.3, 0)
                            st.image(combined_img, caption='Segmentation Mask Overlay', use_column_width=True)
                
                with tab3:
                    st.markdown("### 📈 3D Visualization")
                    if np.any(pred_mask):
                        surface_plot = create_3d_surface_plot(pred_mask)
                        if surface_plot:
                            st.plotly_chart(surface_plot, use_container_width=True)
                    else:
                        st.warning("⚠️ No mask data available for 3D visualization")
            else:
                st.warning("⚠️ No confident detections found. Try adjusting the confidence threshold or using a different image.")
                
    except Exception as e:
        st.error(f"❌ Error during visualization: {str(e)}")
        st.write("Debug: Full error details:", str(e))

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
        
        # Add confidence threshold slider
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.1,
            help="Adjust the confidence threshold for detections"
        )
        # Update model confidence threshold
        if 'model' in st.session_state and st.session_state.model is not None:
            st.session_state.model.conf = conf_threshold
            st.write("Debug: Updated confidence threshold:", conf_threshold)
            
        st.markdown("---")
        st.markdown("""
        ### 📋 Information
        - Model: Deep Learning
        - Classes: Left & Right Ventricle
        - Resolution: 640x640
        """)

    # Main content
    st.title("🫀 3D MRI Heart Analysis Dashboard")
    st.markdown("""
    This advanced dashboard provides comprehensive analysis of heart MRI images using 
    deep learning for segmentation and detection of cardiac structures.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.warning("⚠️ Please ensure you have a stable internet connection and try reloading the page.")
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
