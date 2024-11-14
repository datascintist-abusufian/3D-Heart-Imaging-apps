import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import requests
from torchvision.transforms import transforms
from io import BytesIO
import numpy as np
import cv2
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# --- Configuration ---
st.set_page_config(page_title="3D Heart MRI Analysis", layout="wide", initial_sidebar_state="expanded")

# Constants
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
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
    [data-testid="stSidebarNav"] {
        background-image: url('https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/logo.png');
        background-repeat: no-repeat;
        padding-top: 120px;
        background-position: 20px 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        try:
            if 'model' not in st.session_state:
                response = requests.get(MODEL_URL)
                model_path = BytesIO(response.content)
                st.session_state.model = YOLO(model_path)
            return st.session_state.model
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

def draw_bboxes_and_masks(image, results):
    img = np.array(image)
    confidence_scores = []
    pred_mask = np.zeros((640, 640), dtype=np.uint8)

    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = CLASS_NAMES.get(cls_id, 'Unknown')
            
            confidence_scores.append(conf)
            
            if conf > THRESHOLD:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                if results.masks is not None and i < len(results.masks):
                    mask = results.masks[i].data.cpu().numpy()[0]
                    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                    mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                    pred_mask[y1:y2, x1:x2] = mask_resized

    return img, confidence_scores, pred_mask

def calculate_metrics(true_mask, pred_mask):
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

def create_analysis_metrics(dice, iou, precision, recall, f1):
    cols = st.columns(5)
    
    metrics = [
        ("Dice Score", dice),
        ("IoU Score", iou),
        ("Precision", precision),
        ("Recall", recall),
        ("F1 Score", f1)
    ]
    
    for col, (metric_name, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style='
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            '>
                <h3 style='margin: 0; color: #31333F;'>{metric_name}</h3>
                <p style='font-size: 24px; margin: 10px 0; color: #00a6ed;'>{value:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

def plot_confidence_distribution(confidence_scores):
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
        
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🎯 Detections", "📈 3D View"])
        
        with tab1:
            if np.any(pred_mask):
                ground_truth_mask = np.zeros((640, 640), dtype=np.uint8)
                dice, iou, precision, recall, f1 = calculate_metrics(ground_truth_mask, pred_mask)
                create_analysis_metrics(dice, iou, precision, recall, f1)
                
                conf_fig = plot_confidence_distribution(confidence_scores)
                if conf_fig:
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
            if np.any(pred_mask):
                visualization_3d = create_3d_visualization(pred_mask)
                st.plotly_chart(visualization_3d, use_container_width=True)
            else:
                st.warning("No mask data available for 3D visualization.")

def main():
    # Sidebar
    st.sidebar.title('🎛️ Control Panel')
    st.sidebar.markdown('---')
    
    # Main content
    st.title("🫀 3D MRI Heart Analysis Dashboard")
    st.markdown("""
    This advanced dashboard analyzes heart MRI images using deep learning for cardiac structure segmentation and detection.
    
    ### Features:
    - Real-time heart structure detection
    - Segmentation mask generation
    - 3D visualization
    - Confidence analysis
    - Performance metrics
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
            
            if st.button("🔍 Analyze Image", type="primary"):
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
            
            if st.button("🔍 Analyze Image", type="primary"):
                process_and_analyze(image, model)
                
        except Exception as e:
            st.error(f"Error loading sample image: {e}")

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.markdown("""
    ### 💡 About
    This dashboard is designed for research purposes in cardiac MRI analysis.
    
    **Technologies:**
    - YOLOv5 for detection
    - Streamlit for interface
    - Deep learning for segmentation
    
    **Created by:** Md Abu Sufian  
    **Version:** 2.0.0
    """)

if __name__ == '__main__':
    main()
