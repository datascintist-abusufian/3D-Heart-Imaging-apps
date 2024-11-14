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
THRESHOLD = 0.3

# Page configuration
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
    .analysis-subheader {
        font-size: 1.2rem;
        color: #31333F;
        margin: 0.5rem 0;
    }
    .result-container {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
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

def draw_bboxes_and_masks(image, results):
    """Draw bounding boxes and masks on the image"""
    try:
        img = np.array(image)
        confidence_scores = []
        pred_mask = np.zeros((640, 640), dtype=np.uint8)
        detection_stats = []

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, 'Unknown')
                
                confidence_scores.append(conf)
                detection_stats.append({
                    'Class': label,
                    'Confidence': conf,
                    'Area': (x2-x1)*(y2-y1)
                })
                
                if conf > THRESHOLD:
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add label and confidence score
                    text = f"{label} {conf:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    
                    # Draw text background
                    cv2.rectangle(img, 
                                (text_x, text_y - text_size[1] - 5), 
                                (text_x + text_size[0], text_y + 5), 
                                (255, 0, 0), 
                                -1)
                    
                    # Draw text
                    cv2.putText(img, 
                              text, 
                              (text_x, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, 
                              (255, 255, 255), 
                              2)

                    # Handle segmentation masks
                    if hasattr(results, 'masks') and results.masks is not None and i < len(results.masks):
                        try:
                            mask = results.masks[i].data[0].cpu().numpy()
                            mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                            mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                            pred_mask[y1:y2, x1:x2] = mask_resized
                            
                            # Color overlay
                            roi = img[y1:y2, x1:x2]
                            roi[mask_resized > 127] = (
                                roi[mask_resized > 127] * 0.7 + 
                                np.array([0, 255, 0]) * 0.3
                            ).astype(np.uint8)
                            
                            # Add mask statistics
                            detection_stats[-1].update({
                                'Mask Area': np.sum(mask_resized > 127),
                                'Mask Coverage': np.sum(mask_resized > 127) / ((x2-x1)*(y2-y1))
                            })
                            
                        except Exception as e:
                            st.warning(f"⚠️ Warning: Could not process mask for detection {i}: {str(e)}")

        return img, confidence_scores, pred_mask, detection_stats

    except Exception as e:
        st.error(f"❌ Error in draw_bboxes_and_masks: {str(e)}")
        return image, [], np.zeros((640, 640), dtype=np.uint8), []

def create_analysis_plots(detection_stats):
    """Create various analysis plots"""
    if not detection_stats:
        return None
    
    df = pd.DataFrame(detection_stats)
    
    # Confidence distribution by class
    fig_conf = px.box(df, x='Class', y='Confidence', title='Confidence Distribution by Class')
    fig_conf.update_layout(height=400)
    
    # Area vs Confidence scatter plot
    if 'Mask Area' in df.columns:
        fig_area = px.scatter(df, x='Area', y='Mask Area', 
                            color='Confidence', size='Mask Coverage',
                            title='Detection Area vs Mask Area',
                            labels={'Area': 'Bounding Box Area', 
                                   'Mask Area': 'Segmentation Mask Area'})
        fig_area.update_layout(height=400)
        
        return fig_conf, fig_area
    
    return fig_conf, None

def calculate_advanced_metrics(pred_mask):
    """Calculate advanced metrics for the mask"""
    try:
        if not np.any(pred_mask):
            return {}
        
        # Calculate mask properties
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        metrics = {
            'Number of Regions': len(contours),
            'Total Area': np.sum(pred_mask > 0),
            'Coverage Ratio': np.sum(pred_mask > 0) / pred_mask.size
        }
        
        if contours:
            # Calculate shape metrics for the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            metrics.update({
                'Perimeter': cv2.arcLength(largest_contour, True),
                'Circularity': 4 * np.pi * cv2.contourArea(largest_contour) / 
                              (cv2.arcLength(largest_contour, True) ** 2) if cv2.arcLength(largest_contour, True) > 0 else 0
            })
        
        return metrics
        
    except Exception as e:
        st.error(f"❌ Error calculating advanced metrics: {str(e)}")
        return {}

def create_3d_surface_plot(mask):
    """Create 3D surface plot of segmentation mask"""
    try:
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
            width=None,
            height=600,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        return fig
    except Exception as e:
        st.error(f"❌ Error creating 3D plot: {str(e)}")
        return None

def create_metrics_dashboard(metrics):
    """Create metrics dashboard with styled cards"""
    try:
        cols = st.columns(len(metrics))
        
        metric_styles = {
            'Dice Score': ('🎯', '#FF6B6B'),
            'IoU': ('🔄', '#4ECDC4'),
            'Precision': ('📊', '#45B7D1'),
            'Recall': ('📈', '#96CEB4'),
            'F1 Score': ('⭐', '#FFEEAD'),
            'Number of Regions': ('🔢', '#845EC2'),
            'Total Area': ('📏', '#D65DB1'),
            'Coverage Ratio': ('📊', '#FF6F91'),
            'Perimeter': ('⭕', '#FF9671'),
            'Circularity': ('🔄', '#FFC75F')
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
                    height: 100%;
                '>
                    <h3 style='margin: 0; color: #31333F; font-size: 1rem;'>{icon} {metric_name}</h3>
                    <p style='font-size: 1.5rem; margin: 10px 0; color: {color};'>
                        {value:.3f if isinstance(value, float) else value}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error creating metrics dashboard: {str(e)}")

def process_and_visualize(image, model):
    """Process image and create visualizations"""
    try:
        with st.spinner("🔄 Processing image..."):
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Analysis", 
                "🎯 Segmentation", 
                "📈 3D View",
                "📑 Detailed Metrics"
            ])
            
            # Process image
            img_tensor = process_image(image)
            if img_tensor is None:
                return
            
            # Run inference
            results = model(img_tensor)[0]
            img_with_bboxes, confidence_scores, pred_mask, detection_stats = draw_bboxes_and_masks(image, results)
            
            # Calculate metrics
            if np.any(pred_mask):
                ground_truth_mask = np.zeros((640, 640), dtype=np.uint8)
                dice, iou, precision, recall, f1 = calculate_metrics(ground_truth_mask, pred_mask)
                advanced_metrics = calculate_advanced_metrics(pred_mask)
                
                # Analysis tab
                with tab1:
                    st.markdown("### 📊 Analysis Results")
                    
                    # Basic metrics
                    metrics = {
                        'Dice Score': dice,
                        'IoU': iou,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1
                    }
                    create_metrics_dashboard(metrics)
                    
                    # Detection analysis plots
                    st.markdown("### 📈 Detection Analysis")
                    fig_conf, fig_area = create_analysis_plots(detection_stats)
                    if fig_conf:
                        st.plotly_chart(fig_conf, use_container_width=True)
                    if fig_area:
                        st.plotly_chart(fig_area, use_container_width=True)
                
                # Segmentation tab
                with tab2:
                    st.markdown("### 🎯 Segmentation Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_with_bboxes, caption='Detected Regions', use_column_width=True)
                    with col2:
                        mask_overlay = np.zeros_like(img_with_bboxes)
                        mask_overlay[pred_mask == 255] = (0, 255, 0)
                        combined_img = cv2.addWeighted(img_with_bboxes, 0.7, mask_overlay, 0.3, 0)
                        st.image(combined_img, caption='Segmentation Overlay', use_column_width=True)
                
                # 3D View tab
                with tab3:
                    st.markdown("### 📈 3D Visualization")
                    surface_plot = create_3d_surface_plot(pred_mask)
                    if surface_plot:
                        st.plotly_chart(surface_plot, use_container_width=True)
                
                # Detailed Metrics tab
                with tab4:
                    st.markdown("### 📑 Detailed Analysis")
                    
                    # Advanced metrics
                    st.markdown("#### Advanced Metrics")
                    create_metrics_dashboard(advanced_metrics)
                    
                    # Detection details
                    if detection_stats:
                        st.markdown("#### Detection Details")
                        df = pd.DataFrame(detection_stats)
                        st.dataframe(df, use_container_width=True)
                        
                        # Additional statistics
                        st.markdown("#### Statistical Summary")
                        st.dataframe(df.describe(), use_container_width=True)
            else:
                st.warning("⚠️ No detections found in the image")
                
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
        
        ### 🔍 Analysis Features
        - Ventricle Detection
        - Segmentation Analysis
        - 3D Visualization
        - Advanced Metrics
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
