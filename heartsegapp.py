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
if 'previous_calculations' not in st.session_state:
    st.session_state.previous_calculations = []

@st.cache_resource
def load_model():
    """Load YOLO model with error handling"""
    try:
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model..."):
                st.info("Downloading model... Please wait...")
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")
        
        with st.spinner("🔄 Loading model..."):
            model = YOLO(MODEL_PATH)
            model.conf = CONFIDENCE_THRESHOLD
            model.iou = 0.45
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def process_image(image):
    """Process image for model input"""
    try:
        # Resize image
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0
        
        return img_array, image
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def draw_detections(image, results):
    """Draw bounding boxes on image"""
    try:
        img_draw = np.array(image).copy()
        confidence_scores = []
        detection_stats = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                if conf > CONFIDENCE_THRESHOLD:
                    # Draw box
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    cv2.putText(img_draw, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    confidence_scores.append(conf)
                    detection_stats.append({
                        'Class': CLASS_NAMES[cls_id],
                        'Confidence': conf,
                        'Area': (x2-x1)*(y2-y1)
                    })
        
        return img_draw, confidence_scores, detection_stats
    except Exception as e:
        st.error(f"❌ Error drawing detections: {str(e)}")
        return image, [], []

def create_metrics_dashboard(metrics):
    """Create formatted metrics dashboard"""
    try:
        if not metrics:
            return
            
        cols = st.columns(len(metrics))
        
        for (metric_name, value), col in zip(metrics.items(), cols):
            with col:
                # Format value based on type
                if isinstance(value, (int, np.integer)):
                    formatted_value = f"{value:,}"
                elif isinstance(value, (float, np.floating)):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                # Display metric
                st.metric(
                    label=metric_name,
                    value=formatted_value
                )
                
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")

def create_analysis_report(image_name, metrics, detection_stats):
    """Generate analysis report"""
    try:
        # Convert numpy types to Python types
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            processed_metrics[key] = value
        
        report = {
            "Analysis Report": {
                "Image": image_name,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Metrics": processed_metrics,
                "Detections": detection_stats,
                "Parameters": {
                    "Confidence Threshold": float(CONFIDENCE_THRESHOLD),
                    "Image Size": IMAGE_SIZE
                }
            }
        }
        
        return json.dumps(report, indent=2)
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return None

def process_and_visualize(image, model):
    """Main processing and visualization pipeline"""
    try:
        st.markdown("### Processing Results")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Analysis Results",
            "🎯 Detection View",
            "📈 Visualization"
        ])
        
        # Process image and run model
        with st.spinner("Analyzing image..."):
            # Process image
            processed_img, resized_img = process_image(image)
            if processed_img is None:
                return
            
            # Run inference
            results = model(processed_img)
            
            if len(results) == 0:
                st.warning("No detections found in the image")
                return
            
            # Process detections
            result = results[0]
            img_with_boxes, confidence_scores, detection_stats = draw_detections(resized_img, result)
            
            # Display results in tabs
            with tab1:
                st.markdown("#### Detection Results")
                if detection_stats:
                    # Show metrics
                    metrics = {
                        'Total Detections': len(detection_stats),
                        'Average Confidence': np.mean([d['Confidence'] for d in detection_stats]),
                        'Max Confidence': max([d['Confidence'] for d in detection_stats])
                    }
                    create_metrics_dashboard(metrics)
                    
                    # Show detection details
                    st.markdown("#### Detection Details")
                    df = pd.DataFrame(detection_stats)
                    st.dataframe(df)
                    
                    # Plot confidence distribution
                    fig = px.histogram(
                        df,
                        x='Confidence',
                        nbins=20,
                        title="Confidence Distribution",
                        color_discrete_sequence=['#00a6ed']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create downloadable report
                    report = create_analysis_report("image", metrics, detection_stats)
                    if report:
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.warning("No confident detections found")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Image")
                    st.image(resized_img, caption="Input Image")
                
                with col2:
                    st.markdown("#### Detected Objects")
                    if len(confidence_scores) > 0:
                        st.image(img_with_boxes, caption="Detection Results")
                    else:
                        st.info("No detections to display")
            
            with tab3:
                st.markdown("#### Analysis Visualization")
                if detection_stats:
                    # Create scatter plot
                    fig = px.scatter(
                        df,
                        x='Area',
                        y='Confidence',
                        color='Class',
                        size='Area',
                        title="Detection Analysis",
                        labels={'Area': 'Detection Area', 'Confidence': 'Confidence Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Class distribution
                    class_counts = df['Class'].value_counts()
                    fig2 = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="Class Distribution"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No data available for visualization")

    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))

def main():
    """Main application function"""
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.markdown("---")
        
        # Image source selection
        src = st.radio(
            "📷 Select Image Source",
            ['Upload Image', 'Sample Gallery'],
            help="Choose image source"
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Adjust detection sensitivity"
        )
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Show debug information"
        )
        
        st.markdown("---")
        st.markdown("""
            ### 📋 Model Information
            - Architecture: YOLOv5
            - Task: Object Detection
            - Classes: Heart Ventricles
            - Input Size: 640x640
        """)

    # Main content
    st.title("🫀 3D Heart MRI Analysis Dashboard")
    st.markdown("""
        This advanced dashboard analyzes heart MRI images using deep learning for 
        detection and analysis of cardiac structures.
    """)
    
    # Load model
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("❌ Model loading failed. Please check your connection and reload.")
        return
    
    # Update model confidence threshold
    st.session_state.model.conf = conf_threshold
    
    # Image processing
    try:
        if src == 'Upload Image':
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a heart MRI image"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image', width=300)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    process_and_visualize(image, st.session_state.model)
        else:
            selected_image = st.slider(
                "Select sample image",
                1, 50,
                help="Choose from sample dataset"
            )
            
            try:
                image_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{selected_image}.jpg"
                response = requests.get(image_url)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content)).convert("RGB")
                st.image(image, caption=f'Sample Image #{selected_image}', width=300)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    process_and_visualize(image, st.session_state.model)
            except Exception as e:
                st.error(f"❌ Error loading sample image: {str(e)}")
                if st.session_state.debug_mode:
                    st.write("Debug details:", str(e))

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
