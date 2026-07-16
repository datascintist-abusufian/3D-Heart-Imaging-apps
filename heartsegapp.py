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
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import json
import base64
from scipy import ndimage
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"
MODEL_PATH = "models/yolov8x-seg.pt"
CLASS_NAMES = {0: 'Left Ventricle', 1: 'Right Ventricle'}
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Clinical reference ranges
CLINICAL_REFERENCES = {
    'LVEF': {'normal': (55, 70), 'abnormal': '< 55'},
    'LV/RV Ratio': {'normal': (0.8, 1.2), 'abnormal': 'Outside range'},
    'Ventricle Area': {'normal': 'Varies', 'abnormal': 'Significant asymmetry'}
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced Heart MRI Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #00a6ed;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00a6ed;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .analysis-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1.5rem 0;
        color: #1a1a2e;
        border-bottom: 3px solid #00a6ed;
        padding-bottom: 0.5rem;
    }
    .clinical-indicator {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        margin: 0.2rem 0;
    }
    .normal { background-color: #d4edda; color: #155724; }
    .abnormal { background-color: #f8d7da; color: #721c24; }
    .warning { background-color: #fff3cd; color: #856404; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00a6ed, #0077b6);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,166,237,0.4);
    }
    .sidebar-section {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
def initialize_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'measurements' not in st.session_state:
        st.session_state.measurements = {}

initialize_session_state()

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load YOLO segmentation model with progress tracking"""
    try:
        os.makedirs('models', exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("📥 Downloading segmentation model..."):
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f"Downloading: {progress*100:.1f}%")
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Model downloaded successfully!")
        
        with st.spinner("🔄 Loading model..."):
            model = YOLO(MODEL_PATH)
            model.conf = CONFIDENCE_THRESHOLD
            model.task = 'segment'
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ============================================================================
# IMAGE PROCESSING
# ============================================================================
def process_image(image):
    """Process image for model input with enhanced preprocessing"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Verify image dimensions
        if len(img_array.shape) != 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        if img_array.shape[2] != 3:
            raise ValueError("Image must be RGB")
        
        # Enhanced preprocessing
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Resize image
        resized_image = cv2.resize(enhanced, (IMAGE_SIZE, IMAGE_SIZE))
        
        return resized_image, image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

# ============================================================================
# ADVANCED SEGMENTATION PROCESSING
# ============================================================================
def process_segmentation(image, model):
    """Process segmentation with enhanced feature extraction"""
    try:
        # Run segmentation
        results = model(image, task='segment')
        
        if not results or len(results) == 0:
            return None, None, None
            
        result = results[0]
        
        if not hasattr(result, 'masks') or result.masks is None or len(result.masks) == 0:
            return None, None, None
            
        img_draw = np.array(image).copy()
        
        # Process masks and boxes
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        
        if len(masks) == 0 or len(boxes) == 0:
            return None, None, None
            
        # Create segmentation visualization
        overlay = np.zeros_like(img_draw, dtype=np.uint8)
        stats = []
        
        # Color mapping for different structures
        colors = {
            'Left Ventricle': [0, 255, 0],  # Green
            'Right Ventricle': [255, 0, 0]   # Red
        }
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            try:
                if mask.size == 0:
                    continue
                    
                # Resize mask to image size
                mask = cv2.resize(mask.squeeze(), (image.shape[1], image.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
                
                if not all(np.isfinite(box)):
                    continue
                    
                # Get class and confidence
                cls_id = int(box[5])
                if cls_id not in CLASS_NAMES:
                    continue
                    
                class_name = CLASS_NAMES[cls_id]
                color = colors.get(class_name, [0, 255, 0])
                
                # Create colored overlay
                overlay[mask > 0] = color
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4])
                
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img_draw, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Calculate advanced statistics
                area = np.sum(mask)
                perimeter = cv2.arcLength(mask.astype(np.uint8), True)
                
                # Calculate shape metrics
                if area > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    # Get centroid
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] > 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                    else:
                        cx, cy = 0, 0
                    
                    # Calculate eccentricity using moments
                    if moments['m00'] > 0:
                        mu20 = moments['mu20'] / moments['m00']
                        mu02 = moments['mu02'] / moments['m00']
                        mu11 = moments['mu11'] / moments['m00']
                        angle = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
                        if mu20 + mu02 > 0:
                            eccentricity = np.sqrt((mu20 - mu02)**2 + 4*mu11**2) / (mu20 + mu02)
                        else:
                            eccentricity = 0
                    else:
                        eccentricity = 0
                    
                    # Calculate compactness
                    compactness = area / (perimeter ** 2) if perimeter > 0 else 0
                else:
                    circularity = 0
                    cx, cy = 0, 0
                    eccentricity = 0
                    compactness = 0
                
                stats.append({
                    'Class': class_name,
                    'Confidence': conf,
                    'Area': area,
                    'Perimeter': perimeter,
                    'Circularity': circularity,
                    'Compactness': compactness,
                    'Eccentricity': eccentricity,
                    'Centroid_x': cx,
                    'Centroid_y': cy,
                    'Mask': mask,
                    'Box': [x1, y1, x2, y2]
                })
                
            except Exception as e:
                if st.session_state.debug_mode:
                    st.warning(f"Error processing mask {i}: {str(e)}")
                continue
        
        if not stats:
            return None, None, None
            
        # Combine image with overlay
        alpha = 0.4
        segmented = cv2.addWeighted(img_draw, 1, overlay, alpha, 0)
        
        return segmented, stats, masks
        
    except Exception as e:
        st.error(f"❌ Error in segmentation: {str(e)}")
        return None, None, None

# ============================================================================
# ADVANCED METRICS DASHBOARD
# ============================================================================
def create_metrics_dashboard(stats):
    """Create enhanced analysis dashboard with comprehensive metrics"""
    try:
        if not stats:
            st.warning("No statistics available for analysis")
            return None
            
        # Calculate metrics by class
        metrics_by_class = {}
        for class_name in CLASS_NAMES.values():
            class_stats = [s for s in stats if s['Class'] == class_name]
            if class_stats:
                metrics_by_class[class_name] = {
                    'area': np.mean([s['Area'] for s in class_stats]),
                    'area_std': np.std([s['Area'] for s in class_stats]),
                    'confidence': np.mean([s['Confidence'] for s in class_stats]),
                    'circularity': np.mean([s['Circularity'] for s in class_stats]),
                    'compactness': np.mean([s['Compactness'] for s in class_stats]),
                    'eccentricity': np.mean([s['Eccentricity'] for s in class_stats]),
                    'count': len(class_stats)
                }
        
        # Display summary metrics
        st.markdown("### 📊 Comprehensive Metrics")
        
        cols = st.columns(len(metrics_by_class))
        for idx, (ventricle, metrics) in enumerate(metrics_by_class.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{ventricle}</div>
                    <div class="metric-value">{metrics['area']:,.0f}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">
                        Area (pixels)
                    </div>
                    <hr style="margin: 0.5rem 0;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Confidence</div>
                            <div style="font-weight: bold;">{metrics['confidence']:.2%}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Circularity</div>
                            <div style="font-weight: bold;">{metrics['circularity']:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Compactness</div>
                            <div style="font-weight: bold;">{metrics['compactness']:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Eccentricity</div>
                            <div style="font-weight: bold;">{metrics['eccentricity']:.3f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Spider/Radar Chart
        st.markdown("### 🕸️ Cardiac Function Radar Chart")
        
        # Create radar chart data
        if len(metrics_by_class) >= 2:
            categories = ['Area', 'Circularity', 'Compactness', 'Eccentricity', 'Confidence']
            
            fig_radar = go.Figure()
            
            for ventricle, metrics in metrics_by_class.items():
                values = [
                    metrics['area'] / 1000,  # Normalize area
                    metrics['circularity'],
                    metrics['compactness'] * 10,  # Scale for visibility
                    metrics['eccentricity'],
                    metrics['confidence']
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=ventricle,
                    line=dict(color='#00a6ed' if ventricle == 'Left Ventricle' else '#ff6b6b'),
                    fillcolor='rgba(0,166,237,0.2)' if ventricle == 'Left Ventricle' else 'rgba(255,107,107,0.2)'
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max([max(vals) for vals in [values for values in metrics_by_class.values()]]) * 1.2]
                    )
                ),
                title='Comparative Cardiac Metrics',
                height=500,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Clinical Assessment
        st.markdown("### 🏥 Clinical Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate LV/RV Ratio
            if 'Left Ventricle' in metrics_by_class and 'Right Ventricle' in metrics_by_class:
                lv_area = metrics_by_class['Left Ventricle']['area']
                rv_area = metrics_by_class['Right Ventricle']['area']
                ratio = lv_area / rv_area if rv_area > 0 else 0
                
                # Determine clinical status
                if 0.8 <= ratio <= 1.2:
                    status = "Normal"
                    status_class = "normal"
                elif 0.6 <= ratio <= 1.4:
                    status = "Borderline"
                    status_class = "warning"
                else:
                    status = "Abnormal"
                    status_class = "abnormal"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">LV/RV Ratio</div>
                    <div class="metric-value">{ratio:.2f}</div>
                    <div class="clinical-indicator {status_class}">Status: {status}</div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                        Normal range: 0.8 - 1.2
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Ventricular Asymmetry
            if 'Left Ventricle' in metrics_by_class and 'Right Ventricle' in metrics_by_class:
                lv_area = metrics_by_class['Left Ventricle']['area']
                rv_area = metrics_by_class['Right Ventricle']['area']
                asymmetry = abs(lv_area - rv_area) / ((lv_area + rv_area) / 2) if (lv_area + rv_area) > 0 else 0
                
                if asymmetry < 0.2:
                    status = "Symmetric"
                    status_class = "normal"
                elif asymmetry < 0.4:
                    status = "Mild Asymmetry"
                    status_class = "warning"
                else:
                    status = "Significant Asymmetry"
                    status_class = "abnormal"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Ventricular Asymmetry</div>
                    <div class="metric-value">{asymmetry:.2%}</div>
                    <div class="clinical-indicator {status_class}">Status: {status}</div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                        Lower values indicate better symmetry
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        return metrics_by_class
        
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")
        return None

# ============================================================================
# 3D VISUALIZATION
# ============================================================================
def create_3d_visualization(masks, stats):
    """Create enhanced 3D visualization with multiple views"""
    try:
        if masks is None or len(masks) == 0:
            st.warning("No masks available for 3D visualization")
            return None
            
        # Create tabs for different 3D views
        view_tab1, view_tab2, view_tab3 = st.tabs([
            "🎨 Surface Plot", 
            "🏥 Volume Rendering",
            "📊 Contour Analysis"
        ])
        
        with view_tab1:
            st.markdown("#### Surface Plot Visualization")
            
            # Create subplot for multiple surface views
            fig_surface = make_subplots(
                rows=1, cols=min(2, len(masks)),
                subplot_titles=[f"{CLASS_NAMES.get(i, f'Mask {i+1}')} Surface" 
                               for i in range(min(2, len(masks)))]
            )
            
            for idx, mask in enumerate(masks[:2]):
                z_data = mask.squeeze()
                x, y = np.meshgrid(
                    np.linspace(0, z_data.shape[1], z_data.shape[1]),
                    np.linspace(0, z_data.shape[0], z_data.shape[0])
                )
                
                surface = go.Surface(
                    z=z_data,
                    x=x,
                    y=y,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=(idx == 0)
                )
                
                fig_surface.add_trace(surface, row=1, col=idx+1)
            
            fig_surface.update_layout(
                height=600,
                scene=dict(
                    xaxis_title='Width',
                    yaxis_title='Height',
                    zaxis_title='Intensity',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            )
            st.plotly_chart(fig_surface, use_container_width=True)
        
        with view_tab2:
            st.markdown("#### Volume Rendering")
            
            # Combine masks for volume rendering
            if len(masks) > 0:
                combined_mask = np.zeros_like(masks[0].squeeze())
                for idx, mask in enumerate(masks):
                    combined_mask += (idx + 1) * mask.squeeze()
                
                fig_volume = go.Figure(data=go.Volume(
                    x=np.arange(combined_mask.shape[0]),
                    y=np.arange(combined_mask.shape[1]),
                    z=np.arange(1),
                    value=combined_mask,
                    opacity=0.5,
                    surface_count=20,
                    colorscale='Viridis',
                    showscale=True
                ))
                
                fig_volume.update_layout(
                    title='3D Volume Rendering',
                    scene=dict(
                        xaxis_title='Width',
                        yaxis_title='Height',
                        zaxis_title='Depth',
                        camera=dict(
                            eye=dict(x=2, y=2, z=1.5)
                        )
                    ),
                    height=600
                )
                st.plotly_chart(fig_volume, use_container_width=True)
        
        with view_tab3:
            st.markdown("#### Contour Analysis")
            
            # Create contour plots
            fig_contour = make_subplots(
                rows=1, cols=len(masks),
                subplot_titles=[f"{CLASS_NAMES.get(i, f'Mask {i+1}')} Contours" 
                               for i in range(len(masks))]
            )
            
            for idx, mask in enumerate(masks):
                z_data = mask.squeeze()
                
                contour = go.Contour(
                    z=z_data,
                    colorscale='Viridis',
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True
                    )
                )
                
                fig_contour.add_trace(contour, row=1, col=idx+1)
            
            fig_contour.update_layout(height=500)
            st.plotly_chart(fig_contour, use_container_width=True)
        
        # Add controls
        st.sidebar.markdown("### 🎮 3D Controls")
        opacity = st.sidebar.slider("Opacity", 0.0, 1.0, 0.5, 0.1)
        colormap = st.sidebar.selectbox(
            "Color Scheme",
            ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
        )
        
    except Exception as e:
        st.error(f"❌ Error creating 3D visualization: {str(e)}")
        return None

# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================
def create_time_series_analysis(analysis_history):
    """Create time series analysis if multiple frames available"""
    if len(analysis_history) < 2:
        return None
        
    try:
        st.markdown("### 📈 Temporal Analysis")
        
        # Prepare time series data
        time_data = []
        for i, entry in enumerate(analysis_history):
            if 'stats' in entry:
                for stat in entry['stats']:
                    time_data.append({
                        'Frame': i,
                        'Class': stat['Class'],
                        'Area': stat['Area'],
                        'Confidence': stat['Confidence'],
                        'Circularity': stat.get('Circularity', 0)
                    })
        
        if not time_data:
            return None
            
        df_time = pd.DataFrame(time_data)
        
        # Create subplot
        fig_time = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Area Over Time', 'Confidence Over Time', 'Circularity Over Time'],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Area plot
        for class_name in df_time['Class'].unique():
            subset = df_time[df_time['Class'] == class_name]
            fig_time.add_trace(
                go.Scatter(
                    x=subset['Frame'],
                    y=subset['Area'],
                    mode='lines+markers',
                    name=f'{class_name} - Area',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Confidence plot
        for class_name in df_time['Class'].unique():
            subset = df_time[df_time['Class'] == class_name]
            fig_time.add_trace(
                go.Scatter(
                    x=subset['Frame'],
                    y=subset['Confidence'],
                    mode='lines+markers',
                    name=f'{class_name} - Confidence',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Circularity plot
        for class_name in df_time['Class'].unique():
            subset = df_time[df_time['Class'] == class_name]
            fig_time.add_trace(
                go.Scatter(
                    x=subset['Frame'],
                    y=subset['Circularity'],
                    mode='lines+markers',
                    name=f'{class_name} - Circularity',
                    line=dict(width=2)
                ),
                row=3, col=1
            )
        
        fig_time.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_time, use_container_width=True)
        
        return True
        
    except Exception as e:
        st.warning(f"Could not create time series: {str(e)}")
        return None

# ============================================================================
# REPORT GENERATION
# ============================================================================
def generate_report(stats, metrics, image_info):
    """Generate comprehensive analysis report"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info,
            'metrics': metrics,
            'statistics': stats,
            'clinical_assessment': {}
        }
        
        # Add clinical assessment
        if metrics and 'Left Ventricle' in metrics and 'Right Ventricle' in metrics:
            lv_area = metrics['Left Ventricle']['area']
            rv_area = metrics['Right Ventricle']['area']
            ratio = lv_area / rv_area if rv_area > 0 else 0
            
            report['clinical_assessment']['lv_rv_ratio'] = ratio
            report['clinical_assessment']['ratio_status'] = 'Normal' if 0.8 <= ratio <= 1.2 else 'Abnormal'
        
        # Create download button
        report_json = json.dumps(report, indent=2, default=str)
        
        st.download_button(
            label="📄 Download Detailed Report (JSON)",
            data=report_json,
            file_name=f"heart_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        return report
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return None

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================
def process_and_visualize(image, model):
    """Main processing and visualization pipeline"""
    try:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Segmentation",
            "📊 Analysis Dashboard",
            "📈 3D Visualization",
            "📝 Report"
        ])
        
        # Process image and run segmentation
        with st.spinner("🔄 Processing image..."):
            processed_img, original_img = process_image(image)
            if processed_img is None:
                return
                
            segmented_img, stats, masks = process_segmentation(processed_img, model)
            
            if segmented_img is not None and stats:
                # Store in session state
                st.session_state.current_results = {
                    'stats': stats,
                    'masks': masks,
                    'image': original_img,
                    'segmented': segmented_img
                }
                
                with tab1:
                    st.markdown("### 🎯 Segmentation Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(original_img, caption="Original Image", use_container_width=True)
                        
                        # Image info
                        st.markdown(f"""
                        <div class="metric-card" style="margin-top: 1rem;">
                            <div class="metric-label">Image Information</div>
                            <div>Size: {original_img.size[0]} x {original_img.size[1]} pixels</div>
                            <div>Mode: {original_img.mode}</div>
                            <div>Structures Detected: {len(set([s['Class'] for s in stats]))}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.image(segmented_img, caption="Segmentation Result", use_container_width=True)
                        
                        # Detection summary
                        detection_summary = pd.DataFrame([{
                            'Structure': s['Class'],
                            'Confidence': f"{s['Confidence']:.2%}",
                            'Area': f"{s['Area']:,.0f}",
                            'Circularity': f"{s['Circularity']:.3f}"
                        } for s in stats])
                        
                        st.dataframe(detection_summary, use_container_width=True)
                
                with tab2:
                    st.markdown("### 📊 Analysis Dashboard")
                    metrics = create_metrics_dashboard(stats)
                    
                    # Store in session state
                    st.session_state.measurements = metrics
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'stats': stats,
                        'metrics': metrics
                    })
                
                with tab3:
                    st.markdown("### 📈 3D Visualization")
                    if masks is not None and len(masks) > 0:
                        create_3d_visualization(masks, stats)
                
                with tab4:
                    st.markdown("### 📝 Analysis Report")
                    
                    # Time series analysis
                    if len(st.session_state.analysis_history) > 1:
                        create_time_series_analysis(st.session_state.analysis_history)
                    
                    # Generate report
                    image_info = {
                        'size': original_img.size,
                        'mode': original_img.mode,
                        'format': original_img.format
                    }
                    
                    report = generate_report(stats, metrics, image_info)
                    
                    if report:
                        st.success("✅ Report generated successfully!")
                    else:
                        st.warning("⚠️ Report generation failed")
            else:
                st.warning("⚠️ No segmentation results found in the image")
                
    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application function"""
    st.title("🫀 Advanced Heart MRI Segmentation Analysis")
    st.markdown("""
        This application performs comprehensive segmentation analysis on heart MRI images using 
        advanced deep learning models to identify, quantify, and analyze cardiac structures.
        
        **Features:**
        - 🎯 Precise ventricle segmentation
        - 📊 Comprehensive metrics dashboard
        - 🕸️ Spider/Radar chart analysis
        - 📈 3D visualization
        - 📝 Clinical assessment and reporting
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
            step=0.05,
            help="Minimum confidence score for detection"
        )
        
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode")
        
        # Analysis controls
        st.markdown("### 📊 Analysis Options")
        show_radar = st.checkbox("Show Radar Chart", value=True)
        show_3d = st.checkbox("Show 3D Visualization", value=True)
        
        st.markdown("---")
        st.markdown("""
            ### 📋 Information
            - **Model:** YOLOv8x-seg
            - **Task:** Cardiac Structure Segmentation
            - **Structures:** Left & Right Ventricles
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
                type=["jpg", "jpeg", "png"],
                help="Upload a cardiac MRI image for analysis"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image(image, caption="Uploaded Image", width=300)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    process_and_visualize(image, st.session_state.model)
        else:
            # Sample images
            st.markdown("### 📸 Sample Images")
            
            # Create sample image grid
            cols = st.columns(4)
            sample_images = range(1, 5)
            
            for idx, col in zip(sample_images, cols):
                with col:
                    try:
                        sample_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{idx}.jpg"
                        response = requests.get(sample_url)
                        response.raise_for_status()
                        
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        st.image(image, caption=f"Sample {idx}", width=150)
                        
                        if st.button(f"Analyze #{idx}", key=f"sample_{idx}"):
                            process_and_visualize(image, st.session_state.model)
                    except Exception as e:
                        st.error(f"Error loading sample {idx}")
            
            # Selected sample analysis
            selected_image = st.slider(
                "Or select specific sample",
                1, 10, 1,
                help="Choose from sample dataset"
            )
            
            try:
                sample_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{selected_image}.jpg"
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
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🫀 Advanced Heart MRI Analysis System</p>
            <p style='font-size: 0.8rem;'>For research and educational purposes only | Version 2.0</p>
            <p style='font-size: 0.8rem;'>Created with ❤️ for Medical Imaging Analysis</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
