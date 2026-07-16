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
from skimage import exposure, filters, measure, morphology
from skimage.segmentation import active_contour
from sklearn.cluster import KMeans
from collections import Counter

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
    page_title="Advanced Heart MRI Analysis Suite",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENHANCED CUSTOM CSS
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
    .success-box {
        padding: 1rem;
        background: #d4edda;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background: #d1ecf1;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background: #fff3cd;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
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
    if 'segmentation_attempts' not in st.session_state:
        st.session_state.segmentation_attempts = 0
    if 'failed_images' not in st.session_state:
        st.session_state.failed_images = []

initialize_session_state()

# ============================================================================
# ADVANCED IMAGE ENHANCEMENT
# ============================================================================
def enhance_image(image_array):
    """Apply multiple enhancement techniques"""
    enhanced_versions = []
    
    # Original
    enhanced_versions.append(('Original', image_array.copy()))
    
    # CLAHE
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    clahe_img = cv2.merge([l, a, b])
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2RGB)
    enhanced_versions.append(('CLAHE', clahe_img))
    
    # Histogram equalization
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    hist_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    enhanced_versions.append(('Histogram Equalization', hist_eq))
    
    # Adaptive thresholding for edge enhancement
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    adapt_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    adapt_img = cv2.cvtColor(adapt_thresh, cv2.COLOR_GRAY2RGB)
    enhanced_versions.append(('Adaptive Threshold', adapt_img))
    
    # Unsharp masking
    blurred = cv2.GaussianBlur(image_array, (0, 0), 3)
    unsharp = cv2.addWeighted(image_array, 1.5, blurred, -0.5, 0)
    enhanced_versions.append(('Unsharp Mask', unsharp))
    
    return enhanced_versions

# ============================================================================
# MODEL LOADING WITH FALLBACK
# ============================================================================
@st.cache_resource
def load_model():
    """Load YOLO segmentation model with progress tracking and fallback"""
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
        st.info("💡 Trying to load from cache...")
        try:
            # Try to load from local cache if available
            if os.path.exists(MODEL_PATH):
                model = YOLO(MODEL_PATH)
                return model
        except:
            pass
        return None

# ============================================================================
# ADVANCED SEGMENTATION WITH FALLBACK METHODS
# ============================================================================
def segmentation_with_fallback(image, model):
    """Attempt segmentation with multiple approaches"""
    methods = []
    best_result = None
    highest_confidence = 0
    
    # Method 1: Standard YOLO segmentation
    try:
        results = model(image, task='segment', conf=CONFIDENCE_THRESHOLD)
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                methods.append(('YOLO Segmentation', result, results))
                best_result = result
                highest_confidence = max([box[4] for box in result.boxes.data.cpu().numpy()]) if len(result.boxes) > 0 else 0
    except Exception as e:
        if st.session_state.debug_mode:
            st.warning(f"YOLO segmentation failed: {str(e)}")
    
    # Method 2: YOLO with enhanced preprocessing
    try:
        enhanced_images = enhance_image(np.array(image))
        for name, enhanced_img in enhanced_images[:3]:  # Try top 3 enhancements
            results = model(enhanced_img, task='segment', conf=CONFIDENCE_THRESHOLD * 0.8)
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                    methods.append((f'YOLO + {name}', result, results))
                    avg_conf = np.mean([box[4] for box in result.boxes.data.cpu().numpy()]) if len(result.boxes) > 0 else 0
                    if avg_conf > highest_confidence:
                        best_result = result
                        highest_confidence = avg_conf
    except Exception as e:
        if st.session_state.debug_mode:
            st.warning(f"Enhanced segmentation failed: {str(e)}")
    
    # Method 3: Traditional CV segmentation (fallback)
    if best_result is None:
        try:
            result = traditional_segmentation(image)
            if result is not None:
                methods.append(('Traditional CV', result, None))
                best_result = result
        except Exception as e:
            if st.session_state.debug_mode:
                st.warning(f"Traditional segmentation failed: {str(e)}")
    
    return best_result, methods

# ============================================================================
# TRADITIONAL SEGMENTATION (FALLBACK)
# ============================================================================
def traditional_segmentation(image):
    """Fallback segmentation using traditional CV methods"""
    class TraditionalResult:
        pass
    
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Multi-Otsu thresholding
        thresholds = filters.threshold_multiotsu(gray, classes=3)
        regions = np.digitize(gray, bins=thresholds)
        
        # Find connected components
        labeled = measure.label(regions)
        props = measure.regionprops(labeled)
        
        # Find largest regions (potential ventricles)
        sorted_props = sorted(props, key=lambda x: x.area, reverse=True)[:2]
        
        if len(sorted_props) >= 2:
            # Create pseudo-masks
            masks = []
            boxes = []
            
            for i, prop in enumerate(sorted_props[:2]):
                mask = (labeled == prop.label).astype(np.uint8)
                masks.append(mask)
                
                # Create bounding box
                minr, minc, maxr, maxc = prop.bbox
                box = np.array([minc, minr, maxc, maxr, 0.5, i])  # x1, y1, x2, y2, conf, class
                boxes.append(box)
            
            # Create result object
            result = TraditionalResult()
            result.masks = type('obj', (object,), {'data': np.array(masks)})()
            result.boxes = type('obj', (object,), {'data': np.array(boxes)})()
            result.orig_img = img_array
            
            return result
        
        return None
        
    except Exception as e:
        if st.session_state.debug_mode:
            st.warning(f"Traditional segmentation error: {str(e)}")
        return None

# ============================================================================
# PROCESS SEGMENTATION RESULTS
# ============================================================================
def process_segmentation_results(result, original_img, method_used):
    """Process segmentation results with detailed analysis"""
    try:
        if result is None:
            return None, None, None, None
        
        img_draw = np.array(original_img).copy()
        overlay = np.zeros_like(img_draw, dtype=np.uint8)
        stats = []
        
        # Handle different result types
        if hasattr(result, 'masks') and result.masks is not None:
            # Standard YOLO result
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
        elif hasattr(result, 'masks') and hasattr(result.masks, 'data'):
            # Traditional result
            masks = result.masks.data
            boxes = result.boxes.data
        else:
            # Single mask from traditional segmentation
            masks = [result.masks] if hasattr(result, 'masks') else []
            boxes = [result.boxes] if hasattr(result, 'boxes') else []
        
        if len(masks) == 0:
            return None, None, None, None
            
        colors = {
            'Left Ventricle': [0, 255, 0],
            'Right Ventricle': [255, 0, 0]
        }
        
        for i, mask in enumerate(masks):
            try:
                # Reshape mask if needed
                if len(mask.shape) == 3 and mask.shape[0] > 1:
                    mask = mask[i] if i < len(masks) else mask[0]
                
                # Resize mask to image size
                if mask.shape != (img_draw.shape[0], img_draw.shape[1]):
                    mask = cv2.resize(mask.squeeze(), (img_draw.shape[1], img_draw.shape[0]))
                
                mask = (mask > 0.5).astype(np.uint8)
                
                # Get class info
                if i < len(boxes):
                    box = boxes[i]
                    cls_id = int(box[5]) if len(box) > 5 else i
                    conf = float(box[4]) if len(box) > 4 else 0.5
                else:
                    cls_id = i
                    conf = 0.5
                
                class_name = CLASS_NAMES.get(cls_id, f'Structure_{i}')
                color = colors.get(class_name, [np.random.randint(0, 255) for _ in range(3)])
                
                # Apply mask to overlay
                overlay[mask > 0] = color
                
                # Calculate metrics
                if np.sum(mask) > 0:
                    area = np.sum(mask)
                    perimeter = cv2.arcLength(mask.astype(np.uint8), True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] > 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                    else:
                        cx, cy = 0, 0
                    
                    # Calculate bounding box
                    y_indices, x_indices = np.where(mask > 0)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        x1, y1 = min(x_indices), min(y_indices)
                        x2, y2 = max(x_indices), max(y_indices)
                        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img_draw, f"{class_name} {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        x1, y1, x2, y2 = 0, 0, 0, 0
                    
                    stats.append({
                        'Class': class_name,
                        'Confidence': conf,
                        'Area': area,
                        'Perimeter': perimeter,
                        'Circularity': circularity,
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
            return None, None, None, None
            
        # Combine overlay with original
        alpha = 0.4
        segmented = cv2.addWeighted(img_draw, 1, overlay, alpha, 0)
        
        # Create mask overlay for visualization
        mask_overlay = overlay.copy()
        
        return segmented, stats, masks, mask_overlay
        
    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        return None, None, None, None

# ============================================================================
# ENHANCED METRICS DASHBOARD
# ============================================================================
def create_enhanced_metrics_dashboard(stats, masks):
    """Create comprehensive metrics dashboard with advanced analytics"""
    try:
        if not stats:
            st.warning("No statistics available for analysis")
            return None
            
        # Calculate metrics by class
        metrics_by_class = {}
        for class_name in CLASS_NAMES.values():
            class_stats = [s for s in stats if s['Class'] == class_name]
            if class_stats:
                areas = [s['Area'] for s in class_stats]
                confidences = [s['Confidence'] for s in class_stats]
                circularities = [s['Circularity'] for s in class_stats]
                
                metrics_by_class[class_name] = {
                    'area': np.mean(areas),
                    'area_std': np.std(areas),
                    'area_min': min(areas),
                    'area_max': max(areas),
                    'confidence': np.mean(confidences),
                    'confidence_std': np.std(confidences),
                    'circularity': np.mean(circularities),
                    'circularity_std': np.std(circularities),
                    'count': len(class_stats)
                }
        
        # Display summary metrics with advanced cards
        st.markdown("### 📊 Comprehensive Cardiac Metrics")
        
        # Create 2x2 grid for metrics
        cols = st.columns(2)
        
        for idx, (ventricle, metrics) in enumerate(metrics_by_class.items()):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{ventricle}</div>
                    <div class="metric-value">{metrics['area']:,.0f}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">
                        Area (pixels) ± {metrics['area_std']:,.0f}
                    </div>
                    <hr style="margin: 0.5rem 0;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Confidence</div>
                            <div style="font-weight: bold; color: #28a745;">{metrics['confidence']:.2%}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Circularity</div>
                            <div style="font-weight: bold; color: #007bff;">{metrics['circularity']:.3f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Area Range</div>
                            <div style="font-weight: bold;">{metrics['area_min']:,.0f} - {metrics['area_max']:,.0f}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; color: #6c757d;">Detections</div>
                            <div style="font-weight: bold;">{metrics['count']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Advanced visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Comparative Analysis",
            "🎯 Distribution Plots",
            "🔄 Correlation Matrix",
            "📈 Temporal Trends"
        ])
        
        with tab1:
            st.markdown("#### Comparative Analysis")
            
            # Radar chart
            if len(metrics_by_class) >= 2:
                categories = ['Area', 'Circularity', 'Confidence', 'Compactness']
                fig_radar = go.Figure()
                
                for ventricle, metrics in metrics_by_class.items():
                    values = [
                        metrics['area'] / 1000,
                        metrics['circularity'],
                        metrics['confidence'],
                        metrics['circularity'] * 10
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
                    showlegend=True
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        with tab2:
            st.markdown("#### Distribution Plots")
            
            # Create distribution plots for each metric
            fig_dist = make_subplots(rows=2, cols=2, 
                                    subplot_titles=['Area Distribution', 'Confidence Distribution',
                                                  'Circularity Distribution', 'Perimeter Distribution'])
            
            for i, (metric, title) in enumerate([('Area', 'Area'), ('Confidence', 'Confidence'),
                                               ('Circularity', 'Circularity'), ('Perimeter', 'Perimeter')]):
                row = i // 2 + 1
                col = i % 2 + 1
                
                for ventricle in metrics_by_class.keys():
                    values = [s[metric] for s in stats if s['Class'] == ventricle]
                    if values:
                        fig_dist.add_trace(
                            go.Histogram(x=values, name=ventricle, opacity=0.7, 
                                       marker_color='#00a6ed' if ventricle == 'Left Ventricle' else '#ff6b6b'),
                            row=row, col=col
                        )
            
            fig_dist.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab3:
            st.markdown("#### Correlation Matrix")
            
            # Create correlation matrix
            df = pd.DataFrame(stats)
            numeric_cols = ['Area', 'Confidence', 'Circularity']
            if all(col in df.columns for col in numeric_cols):
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='Feature Correlation Matrix',
                    height=500
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            st.markdown("#### Temporal Analysis")
            if st.session_state.analysis_history:
                create_temporal_analysis()
            else:
                st.info("Process more images to see temporal trends")
        
        # Clinical assessment
        st.markdown("### 🏥 Clinical Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # LV/RV Ratio
            if 'Left Ventricle' in metrics_by_class and 'Right Ventricle' in metrics_by_class:
                lv_area = metrics_by_class['Left Ventricle']['area']
                rv_area = metrics_by_class['Right Ventricle']['area']
                ratio = lv_area / rv_area if rv_area > 0 else 0
                
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
        
        with col3:
            # Overall Confidence Score
            avg_confidence = np.mean([s['Confidence'] for s in stats])
            if avg_confidence > 0.8:
                status = "High Confidence"
                status_class = "normal"
            elif avg_confidence > 0.6:
                status = "Medium Confidence"
                status_class = "warning"
            else:
                status = "Low Confidence"
                status_class = "abnormal"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Confidence</div>
                <div class="metric-value">{avg_confidence:.2%}</div>
                <div class="clinical-indicator {status_class}">Status: {status}</div>
                <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                    Based on all detected structures
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        return metrics_by_class
        
    except Exception as e:
        st.error(f"❌ Error creating dashboard: {str(e)}")
        return None

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================
def create_temporal_analysis():
    """Create temporal analysis from history"""
    try:
        history = st.session_state.analysis_history[-10:]  # Last 10 frames
        
        # Extract data
        time_data = []
        for i, entry in enumerate(history):
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
            return
        
        df = pd.DataFrame(time_data)
        
        # Create trend plots
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=['Area Trends', 'Confidence Trends', 'Circularity Trends'],
                           shared_xaxes=True)
        
        for class_name in df['Class'].unique():
            subset = df[df['Class'] == class_name]
            
            fig.add_trace(
                go.Scatter(x=subset['Frame'], y=subset['Area'],
                          mode='lines+markers', name=f'{class_name} - Area',
                          line=dict(width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=subset['Frame'], y=subset['Confidence'],
                          mode='lines+markers', name=f'{class_name} - Confidence',
                          line=dict(width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=subset['Frame'], y=subset['Circularity'],
                          mode='lines+markers', name=f'{class_name} - Circularity',
                          line=dict(width=2)),
                row=3, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not create temporal analysis: {str(e)}")

# ============================================================================
# ENHANCED 3D VISUALIZATION
# ============================================================================
def create_enhanced_3d_visualization(masks, stats):
    """Create enhanced 3D visualization with multiple views"""
    try:
        if masks is None or len(masks) == 0:
            st.warning("No masks available for 3D visualization")
            return None
            
        tabs = st.tabs([
            "🎨 3D Surface",
            "🏥 Volume Rendering",
            "📊 Contour Analysis",
            "🔄 Dynamic View"
        ])
        
        with tabs[0]:
            st.markdown("#### 3D Surface Visualization")
            
            fig = make_subplots(rows=1, cols=min(3, len(masks)),
                               subplot_titles=[f"{CLASS_NAMES.get(i, f'Mask {i+1}')} Surface" 
                                              for i in range(min(3, len(masks)))])
            
            for idx, mask in enumerate(masks[:3]):
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
                
                fig.add_trace(surface, row=1, col=idx+1)
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.markdown("#### Volume Rendering")
            
            if len(masks) > 0:
                combined = np.zeros_like(masks[0].squeeze())
                for idx, mask in enumerate(masks):
                    combined += (idx + 1) * mask.squeeze()
                
                fig = go.Figure(data=go.Volume(
                    x=np.arange(combined.shape[0]),
                    y=np.arange(combined.shape[1]),
                    z=np.arange(1),
                    value=combined,
                    opacity=0.5,
                    surface_count=20,
                    colorscale='Viridis'
                ))
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown("#### Contour Analysis")
            
            fig = make_subplots(rows=1, cols=len(masks),
                               subplot_titles=[f"{CLASS_NAMES.get(i, f'Mask {i+1}')} Contours" 
                                              for i in range(len(masks))])
            
            for idx, mask in enumerate(masks):
                z_data = mask.squeeze()
                
                contour = go.Contour(
                    z=z_data,
                    colorscale='Viridis',
                    contours=dict(coloring='heatmap', showlabels=True)
                )
                
                fig.add_trace(contour, row=1, col=idx+1)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.markdown("#### Dynamic 3D View")
            
            # Create animated 3D visualization
            fig = go.Figure()
            
            for idx, mask in enumerate(masks):
                z_data = mask.squeeze()
                x, y = np.meshgrid(
                    np.linspace(0, z_data.shape[1], z_data.shape[1]),
                    np.linspace(0, z_data.shape[0], z_data.shape[0])
                )
                
                fig.add_trace(go.Surface(
                    z=z_data,
                    x=x,
                    y=y,
                    colorscale='Viridis',
                    opacity=0.7,
                    name=CLASS_NAMES.get(idx, f'Structure {idx+1}')
                ))
            
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    xaxis_title='Width',
                    yaxis_title='Height',
                    zaxis_title='Intensity'
                ),
                height=600,
                title='Interactive 3D View'
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating 3D visualization: {str(e)}")

# ============================================================================
# ENHANCED REPORT GENERATION
# ============================================================================
def generate_enhanced_report(stats, metrics, image_info, methods_used):
    """Generate comprehensive analysis report with all details"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_info': image_info,
            'methods_tried': methods_used,
            'detected_structures': len(stats),
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
            report['clinical_assessment']['asymmetry'] = abs(lv_area - rv_area) / ((lv_area + rv_area) / 2) if (lv_area + rv_area) > 0 else 0
        
        # Generate PDF-like report using markdown
        report_md = f"""
# 🫀 Cardiac MRI Analysis Report

## 📋 Summary
- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Image Size:** {image_info.get('size', 'N/A')}
- **Detected Structures:** {len(stats)}
- **Methods Used:** {', '.join(methods_used)}

## 📊 Metrics
"""
        
        for ventricle, metrics in metrics.items() if metrics else {}:
            report_md += f"""
### {ventricle}
- **Area:** {metrics['area']:,.0f} ± {metrics.get('area_std', 0):,.0f} pixels
- **Confidence:** {metrics['confidence']:.2%}
- **Circularity:** {metrics['circularity']:.3f}
- **Detections:** {metrics['count']}
"""
        
        if 'clinical_assessment' in report:
            report_md += f"""
## 🏥 Clinical Assessment
- **LV/RV Ratio:** {report['clinical_assessment'].get('lv_rv_ratio', 0):.2f} ({report['clinical_assessment'].get('ratio_status', 'Unknown')})
- **Asymmetry:** {report['clinical_assessment'].get('asymmetry', 0):.2%}
"""
        
        # Create download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download JSON Report",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"heart_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📝 Download Markdown Report",
                data=report_md,
                file_name=f"heart_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        return report
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return None

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================
def process_and_visualize_enhanced(image, model):
    """Enhanced main processing and visualization pipeline"""
    try:
        # Create tabs
        tabs = st.tabs([
            "🎯 Segmentation",
            "📊 Advanced Dashboard",
            "📈 3D Visualization",
            "📝 Comprehensive Report",
            "🔍 Quality Analysis"
        ])
        
        # Process image with multiple methods
        with st.spinner("🔄 Processing image with multiple methods..."):
            # Try different enhancement techniques
            enhanced_images = enhance_image(np.array(image))
            
            best_result = None
            best_method = None
            best_stats = None
            best_segmented = None
            best_masks = None
            best_overlay = None
            methods_used = []
            
            # Try each enhancement
            for name, enhanced_img in enhanced_images:
                if st.session_state.debug_mode:
                    st.info(f"Trying {name}...")
                
                # Run segmentation
                result, methods = segmentation_with_fallback(Image.fromarray(enhanced_img), model)
                methods_used.extend([m[0] for m in methods])
                
                if result is not None:
                    # Process results
                    segmented, stats, masks, overlay = process_segmentation_results(
                        result, Image.fromarray(enhanced_img), name
                    )
                    
                    if stats and len(stats) > 0:
                        # Check if this is better
                        avg_conf = np.mean([s['Confidence'] for s in stats])
                        if best_result is None or avg_conf > np.mean([s['Confidence'] for s in best_stats]) if best_stats else 0:
                            best_result = result
                            best_method = name
                            best_stats = stats
                            best_segmented = segmented
                            best_masks = masks
                            best_overlay = overlay
            
            # If no results from enhanced methods, try one more time with original
            if best_result is None:
                result, methods = segmentation_with_fallback(image, model)
                methods_used.extend([m[0] for m in methods])
                
                if result is not None:
                    segmented, stats, masks, overlay = process_segmentation_results(
                        result, image, 'Original'
                    )
                    if stats and len(stats) > 0:
                        best_result = result
                        best_method = 'Original'
                        best_stats = stats
                        best_segmented = segmented
                        best_masks = masks
                        best_overlay = overlay
        
        if best_result is not None and best_stats:
            # Store results
            st.session_state.current_results = {
                'stats': best_stats,
                'masks': best_masks,
                'image': image,
                'segmented': best_segmented,
                'method': best_method,
                'overlay': best_overlay
            }
            
            with tabs[0]:
                st.markdown(f"### 🎯 Segmentation Results (Method: {best_method})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                    
                    # Show enhancement methods tried
                    with st.expander("🔍 Methods Attempted"):
                        for method in set(methods_used):
                            st.write(f"• {method}")
                
                with col2:
                    if best_segmented is not None:
                        st.image(best_segmented, caption="Segmentation Result", use_container_width=True)
                    
                    # Detection summary
                    df_summary = pd.DataFrame([{
                        'Structure': s['Class'],
                        'Confidence': f"{s['Confidence']:.2%}",
                        'Area': f"{s['Area']:,.0f}",
                        'Circularity': f"{s['Circularity']:.3f}"
                    } for s in best_stats])
                    
                    st.dataframe(df_summary, use_container_width=True)
            
            with tabs[1]:
                st.markdown("### 📊 Advanced Analytics Dashboard")
                metrics = create_enhanced_metrics_dashboard(best_stats, best_masks)
                
                # Store in session state
                st.session_state.measurements = metrics
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'stats': best_stats,
                    'metrics': metrics,
                    'method': best_method
                })
            
            with tabs[2]:
                st.markdown("### 📈 3D Visualization")
                if best_masks is not None and len(best_masks) > 0:
                    create_enhanced_3d_visualization(best_masks, best_stats)
            
            with tabs[3]:
                st.markdown("### 📝 Comprehensive Report")
                
                # Temporal analysis if available
                if len(st.session_state.analysis_history) > 1:
                    create_temporal_analysis()
                
                # Generate report
                image_info = {
                    'size': image.size,
                    'mode': image.mode,
                    'format': image.format
                }
                
                report = generate_enhanced_report(best_stats, metrics, image_info, 
                                                list(set(methods_used)))
                
                if report:
                    st.success("✅ Report generated successfully!")
            
            with tabs[4]:
                st.markdown("### 🔍 Quality Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Image Quality Metrics")
                    
                    # Calculate image quality metrics
                    img_array = np.array(image)
                    
                    # Contrast
                    contrast = np.std(img_array)
                    st.metric("Contrast", f"{contrast:.2f}")
                    
                    # Brightness
                    brightness = np.mean(img_array)
                    st.metric("Brightness", f"{brightness:.2f}")
                    
                    # Sharpness (Laplacian variance)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    st.metric("Sharpness", f"{sharpness:.2f}")
                
                with col2:
                    st.markdown("#### Segmentation Quality")
                    
                    # Detection coverage
                    total_area = image.size[0] * image.size[1]
                    detected_area = sum([s['Area'] for s in best_stats])
                    coverage = detected_area / total_area if total_area > 0 else 0
                    st.metric("Coverage", f"{coverage:.2%}")
                    
                    # Detection count
                    st.metric("Structures Detected", len(best_stats))
                    
                    # Average confidence
                    avg_conf = np.mean([s['Confidence'] for s in best_stats])
                    st.metric("Avg Confidence", f"{avg_conf:.2%}")
                
                # Quality recommendation
                st.markdown("#### 💡 Recommendations")
                
                recommendations = []
                if contrast < 30:
                    recommendations.append("Consider image contrast enhancement")
                if avg_conf < 0.6:
                    recommendations.append("Try a different image or adjust confidence threshold")
                if coverage < 0.1:
                    recommendations.append("Image may need preprocessing or different segmentation approach")
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(f"• {rec}")
                else:
                    st.success("✅ Image quality appears good for analysis!")
        
        else:
            st.error("❌ No structures detected in the image")
            st.info("💡 Try adjusting the confidence threshold or using a different image")
            
            # Show debug info
            if st.session_state.debug_mode:
                st.write("Methods attempted:", methods_used)
                st.write("Number of attempts:", len(methods_used))
            
            # Increment failure counter
            st.session_state.segmentation_attempts += 1
            
    except Exception as e:
        st.error(f"❌ Error in processing: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Enhanced main application function"""
    st.markdown("""
    <div class="gradient-header">
        <h1>🫀 Advanced Heart MRI Analysis Suite</h1>
        <p>Comprehensive segmentation and analysis of cardiac structures</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.error("❌ Model loading failed. Please check your connection.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Image source
        src = st.radio(
            "📷 Select Image Source",
            ['Upload Image', 'Sample Images', 'URL'],
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
        st.session_state.model.conf = conf_threshold
        
        # Advanced options
        st.markdown("### 🎯 Advanced Options")
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode")
        use_enhancement = st.checkbox("✨ Use Image Enhancement", value=True)
        
        # Analysis controls
        st.markdown("### 📊 Analysis Options")
        show_radar = st.checkbox("Show Radar Chart", value=True)
        show_3d = st.checkbox("Show 3D Visualization", value=True)
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Images Processed", len(st.session_state.analysis_history))
        st.metric("Failed Attempts", st.session_state.segmentation_attempts)
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.session_state.segmentation_attempts = 0
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
            ### 📋 Information
            - **Model:** YOLOv8x-seg
            - **Task:** Cardiac Structure Segmentation
            - **Structures:** Left & Right Ventricles
            - **Version:** 3.0 (Enhanced)
        """)
    
    # Main content
    try:
        if src == 'Upload Image':
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Upload a cardiac MRI image for analysis"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    process_and_visualize_enhanced(image, st.session_state.model)
        
        elif src == 'Sample Images':
            st.markdown("### 📸 Sample Images")
            
            # Sample image grid with better layout
            sample_urls = [
                "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/data/images/test/1.jpg",
                "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/data/images/test/2.jpg",
                "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/data/images/test/3.jpg",
                "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/data/images/test/4.jpg"
            ]
            
            cols = st.columns(4)
            for idx, (col, url) in enumerate(zip(cols, sample_urls)):
                with col:
                    try:
                        response = requests.get(url, timeout=5)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        st.image(image, caption=f"Sample {idx+1}", use_container_width=True)
                        
                        if st.button(f"Analyze #{idx+1}", key=f"sample_{idx}"):
                            process_and_visualize_enhanced(image, st.session_state.model)
                    except Exception as e:
                        st.error(f"Error loading sample {idx+1}")
        
        else:  # URL
            url = st.text_input("Enter Image URL:", placeholder="https://example.com/image.jpg")
            if url:
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    st.image(image, caption="Image from URL", use_container_width=True)
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        process_and_visualize_enhanced(image, st.session_state.model)
                except Exception as e:
                    st.error(f"❌ Error loading image from URL: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error in application: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Debug details:", str(e))
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🫀 Advanced Heart MRI Analysis System v3.0</p>
            <p style='font-size: 0.8rem;'>For research and educational purposes only</p>
            <p style='font-size: 0.8rem;'>Created with ❤️ for Medical Imaging Analysis</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
