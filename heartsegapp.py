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
    .alert {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-info { background-color: #e3f2fd; }
    .alert-warning { background-color: #fff3cd; }
    .alert-error { background-color: #f8d7da; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'previous_calculations' not in st.session_state:
    st.session_state.previous_calculations = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

@st.cache_resource
def load_model():
    """Load and configure YOLO model with error handling"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Download model if not exists
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Downloading model..."):
                st.info("Initializing model download. Please wait...")
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")
        
        # Load and configure model
        with st.spinner("🔄 Loading model..."):
            model = YOLO(MODEL_PATH)
            model.conf = CONFIDENCE_THRESHOLD  # Set confidence threshold
            model.iou = 0.45  # Set IoU threshold
            model.max_det = 100  # Maximum detections
            model.task = 'detect'
            st.success("✅ Model loaded successfully!")
            return model
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error while downloading model: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
    return None

def process_image(image):
    """Process and prepare image for model inference"""
    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        return img_array, image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def draw_detections(image, results):
    """Draw bounding boxes and create detection visualization"""
    try:
        # Create copy of image for drawing
        img_draw = np.array(image).copy()
        confidence_scores = []
        detection_stats = []
        
        # Create empty mask
        pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        
        if not hasattr(results, 'boxes') or results.boxes is None:
            return img_draw, [], pred_mask, []
            
        # Process each detection
        for i, box in enumerate(results.boxes):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            if conf > CONFIDENCE_THRESHOLD:
                # Draw bounding box
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.putText(img_draw, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add to confidence scores
                confidence_scores.append(conf)
                
                # Add detection statistics
                detection_stats.append({
                    'Class': CLASS_NAMES[cls_id],
                    'Confidence': conf,
                    'Area': (x2-x1)*(y2-y1),
                    'Center X': (x1+x2)/2,
                    'Center Y': (y1+y2)/2
                })
                
                # Handle segmentation mask if available
                if hasattr(results, 'masks') and results.masks is not None:
                    try:
                        mask = results.masks[i].data.cpu().numpy()
                        mask = cv2.resize(mask, (x2-x1, y2-y1))
                        mask = (mask > 0.5).astype(np.uint8) * 255
                        pred_mask[y1:y2, x1:x2] = mask
                    except Exception as e:
                        if st.session_state.debug_mode:
                            st.warning(f"Mask processing error: {str(e)}")
        
        return img_draw, confidence_scores, pred_mask, detection_stats
        
    except Exception as e:
        st.error(f"❌ Error drawing detections: {str(e)}")
        return image, [], np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8), []

def create_visualization(image, results):
    """Create comprehensive visualization of results"""
    try:
        # Draw detections
        img_with_boxes, confidence_scores, pred_mask, detection_stats = draw_detections(image, results)
        
        # Create visualizations
        visualizations = {}
        
        # Original with detections
        visualizations['detections'] = img_with_boxes
        
        # Segmentation mask overlay if available
        if np.any(pred_mask):
            mask_overlay = np.zeros_like(img_with_boxes)
            mask_overlay[pred_mask == 255] = [0, 255, 0]  # Green overlay
            overlay = cv2.addWeighted(img_with_boxes, 0.7, mask_overlay, 0.3, 0)
            visualizations['segmentation'] = overlay
        
        # Confidence distribution
        if confidence_scores:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=confidence_scores,
                nbinsx=20,
                marker_color='#00a6ed'
            ))
            fig.update_layout(
                title='Detection Confidence Distribution',
                xaxis_title='Confidence',
                yaxis_title='Count',
                height=400
            )
            visualizations['confidence_dist'] = fig
        
        return visualizations, detection_stats
        
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")
        return {}, []

def calculate_metrics(detections, mask):
    """Calculate comprehensive analysis metrics"""
    try:
        metrics = {}
        
        # Detection metrics
        if detections:
            metrics.update({
                'Total Detections': len(detections),
                'Average Confidence': np.mean([d['Confidence'] for d in detections]),
                'Max Confidence': max([d['Confidence'] for d in detections]),
                'Average Area': np.mean([d['Area'] for d in detections])
            })
            
            # Class distribution
            class_counts = {}
            for d in detections:
                class_counts[d['Class']] = class_counts.get(d['Class'], 0) + 1
            metrics['Class Distribution'] = class_counts
        
        # Mask metrics
        if np.any(mask):
            # Calculate mask properties
            total_pixels = mask.size
            mask_pixels = np.sum(mask > 0)
            
            metrics.update({
                'Coverage (%)': (mask_pixels / total_pixels) * 100,
                'Total Area (pixels)': mask_pixels
            })
            
            # Calculate contour properties
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                metrics.update({
                    'Number of Regions': len(contours),
                    'Perimeter': cv2.arcLength(largest_contour, True),
                    'Compactness': (4 * np.pi * cv2.contourArea(largest_contour)) / 
                                 (cv2.arcLength(largest_contour, True) ** 2)
                })
        
        return metrics
        
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        return {}

def create_metrics_dashboard(metrics):
    """Create interactive metrics dashboard"""
    try:
        if not metrics:
            return
        
        # Create metric cards
        cols = st.columns(3)
        
        # Define metric styles
        metric_styles = {
            'Total Detections': ('🎯', '#FF6B6B'),
            'Average Confidence': ('📊', '#4ECDC4'),
            'Coverage (%)': ('📈', '#45B7D1'),
            'Number of Regions': ('🔍', '#96CEB4'),
            'Total Area (pixels)': ('📏', '#FFEEAD')
        }
        
        # Display metrics in cards
        for i, (metric_name, value) in enumerate(metrics.items()):
            if metric_name == 'Class Distribution':
                continue  # Handle separately
                
            with cols[i % 3]:
                icon, color = metric_styles.get(metric_name, ('📌', '#666666'))
                st.markdown(f"""
                    <div style='
                        background-color: {color}22;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        border: 2px solid {color};
                    '>
                        <h3 style='margin: 0; color: #31333F; font-size: 1rem;'>
                            {icon} {metric_name}
                        </h3>
                        <p class='metric-value' style='color: {color};'>
                            {value:.2f if isinstance(value, float) else value}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Display class distribution if available
        if 'Class Distribution' in metrics:
            st.markdown("### 📊 Class Distribution")
            df = pd.DataFrame(list(metrics['Class Distribution'].items()),
                            columns=['Class', 'Count'])
            
            fig = px.bar(df, x='Class', y='Count',
                        color='Class',
                        title="Detections by Class")
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")

def create_3d_visualization(mask):
    """Create 3D surface plot of segmentation mask"""
    try:
        if not np.any(mask):
            return None
            
        # Create meshgrid for 3D plot
        x, y = np.meshgrid(
            np.linspace(0, mask.shape[1], mask.shape[1]),
            np.linspace(0, mask.shape[0], mask.shape[0])
        )
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=mask, x=x, y=y)])
        
        # Update layout
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
        st.error(f"❌ Error creating 3D visualization: {str(e)}")
        return None

def create_analysis_report(image_name, metrics, timestamp):
    """Generate analysis report"""
    try:
        report = {
            "Analysis Report": {
                "Image": image_name,
                "Date": timestamp,
                "Metrics": metrics,
                "Analysis Parameters": {
                    "Confidence Threshold": CONFIDENCE_THRESHOLD,
                    "Image Size": IMAGE_SIZE,
                    "Model": "YOLOv5"
                }
            }
        }
        
        return json.dumps(report, indent=2)
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return None

def display_results(image, visualizations, metrics, detection_stats):
    """Display analysis results in organized tabs"""
    try:
        tab1, tab2, tab3 = st.tabs([
            "📊 Analysis Results",
            "🎯 Detections View",
            "📈 3D Visualization"
        ])
        
        with tab1:
            st.markdown("### 📊 Analysis Dashboard")
            create_metrics_dashboard(metrics)
            
            if detection_stats:
                st.markdown("### 📋 Detection Details")
                df = pd.DataFrame(detection_stats)
                st.dataframe(df)
                
                if 'confidence_dist' in visualizations:
                    st.plotly_chart(visualizations['confidence_dist'],
                                  use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(image, caption='Input Image', use_column_width=True)
            
            with col2:
                st.markdown("#### Detection Results")
                if 'detections' in visualizations:
                    st.image(visualizations['detections'],
                            caption='Detected Regions',
                            use_column_width=True)
                
                if 'segmentation' in visualizations:
                    st.markdown("#### Segmentation Overlay")
                    st.image(visualizations['segmentation'],
                            caption='Segmentation Mask',
                            use_column_width=True)
        
        with tab3:
            st.markdown("### 📈 3D Visualization")
            surface_plot = create_3d_visualization(metrics.get('mask', None))
            if surface_plot:
                st.plotly_chart(surface_plot, use_container_width=True)
            else:
                st.info("3D visualization not available for this image")
                
    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")

def main():
    """Main application function"""
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        st.markdown("---")
        
        # Image source selection
        src = st.radio(
            "📷 Select Image Source",
            ['Sample Gallery', 'Upload Image'],
            help="Choose image source"
        )
        
        # Confidence threshold adjustment
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Adjust detection sensitivity"
        )
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Show additional debugging information"
        )
        
        st.markdown("---")
        st.markdown("""
            ### 📋 Model Information
            - Architecture: YOLOv5
            - Task: Detection & Segmentation
            - Classes: Left & Right Ventricle
            - Input Size: 640x640
        """)

    # Main content
    st.title("🫀 3D Heart MRI Analysis Dashboard")
    st.markdown("""
        This advanced dashboard provides comprehensive analysis of heart MRI images
        using deep learning for detection and segmentation of cardiac structures.
    """)

    # Load model
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("❌ Model loading failed. Please check your connection and reload.")
        return

    # Update model confidence threshold
    st.session_state.model.conf = conf_threshold
    if st.session_state.debug_mode:
        st.write(f"Debug: Confidence threshold set to {conf_threshold}")

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
                image_name = uploaded_file.name
        else:
            selected_image = st.slider(
                "Select sample image",
                1, 50,
                help="Choose from sample dataset"
            )
            
            image_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{selected_image}.jpg"
            response = requests.get(image_url)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption=f'Sample Image #{selected_image}', width=300)
            image_name = f"sample_{selected_image}.jpg"

        # Analysis button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing image..."):
                # Process image
                processed_img, resized_img = process_image(image)
                if processed_img is None:
                    st.error("❌ Error processing image")
                    return

                # Run inference
                if st.session_state.debug_mode:
                    st.write("Debug: Running model inference")
                results = st.session_state.model(processed_img)
                
                if len(results) > 0:
                    # Create visualizations
                    visualizations, detection_stats = create_visualization(
                        resized_img, results[0]
                    )
                    
                    # Calculate metrics
                    pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                    if hasattr(results[0], 'masks') and results[0].masks is not None:
                        pred_mask = results[0].masks[0].data.cpu().numpy()
                    
                    metrics = calculate_metrics(detection_stats, pred_mask)
                    metrics['mask'] = pred_mask
                    
                    # Display results
                    display_results(resized_img, visualizations, metrics, detection_stats)
                    
                    # Generate report
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    report = create_analysis_report(image_name, metrics, timestamp)
                    
                    if report:
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"analysis_report_{timestamp}.json",
                            mime="application/json"
                        )
                    
                    # Store in session state
                    st.session_state.previous_calculations.append({
                        "timestamp": timestamp,
                        "image_name": image_name,
                        "metrics": metrics,
                        "detection_stats": detection_stats
                    })
                else:
                    st.warning("⚠️ No detections found in the image")

    except Exception as e:
        st.error(f"❌ Error in main application: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug: Full error details:", str(e))

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
