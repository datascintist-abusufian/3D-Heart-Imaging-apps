import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
from torchvision.transforms import transforms
from io import BytesIO
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


# --- Configuration ---
MODEL_PATH = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/blob/main/yolov5s.pt"
GIF_PATH = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/blob/main/WholeHeartSegment_ErrorMap_WhiteBg.gif"
SAMPLE_IMAGES_DIR = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/tree/main/data/images/test"
LABELS_DIR = SAMPLE_IMAGES_DIR  # Assuming labels are in the same directory
CLASS_NAMES = {0: 'left ventricle', 1: 'right ventricle'}
THRESHOLD = 0.5  # Confidence threshold for considering a detection as correct

# Ground truth data file path (assuming it's a JSON file)
ground_truth_path = "annotations.json"

@st.cache_resource
def load_model():
    model_file = "yolov5s.pt"
    if not os.path.exists(model_file):
        st.write("Downloading model...")
        response = requests.get(MODEL_PATH)
        with open(model_file, 'wb') as f:
            f.write(response.content)
        st.write("Model downloaded successfully.")
    
    try:
        st.write("Loading model from path...")
        model = YOLO(model_file)  # Load YOLOv5 model
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    return model

def load_ground_truth():
    if not os.path.exists(ground_truth_path):
        st.error(f"Ground truth file not found at {ground_truth_path}")
        return None

    try:
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)
        st.write("Ground truth data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading ground truth data: {e}")
        return None

    return gt_data

def process_image(image):
    st.write("Processing image...")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust to the input size your model expects
        transforms.ToTensor(),
    ])

    try:
        image = transform(image).unsqueeze(0)
        st.write("Image processed successfully.")
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def draw_bboxes(image, results, ground_truth=None, image_name=None):
    img = np.array(image)
    class_names = {0: 'left ventricle', 1: 'right ventricle'}  # Assuming these are your class indices
    
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls_id = result.int().tolist()
        label = class_names.get(cls_id, 'Unknown')
        score = conf

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{label} {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (255, 0, 0), -1)
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if ground_truth is not None and image_name in ground_truth:
        gt_records = ground_truth[image_name]
        for gt in gt_records:
            x1, y1, x2, y2, cls_id = gt['x1'], gt['y1'], gt['x2'], gt['y2'], gt['cls_id']
            label = class_names.get(cls_id, 'Unknown')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img

def calculate_iou(pred_box, gt_box):
    x1_p, y1_p, x2_p, y2_p = pred_box
    x1_g, y1_g, x2_g, y2_g = gt_box

    xi1 = max(x1_p, x1_g)
    yi1 = max(y1_p, y1_g)
    xi2 = min(x2_p, x2_g)
    yi2 = min(y2_p, y2_g)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    box2_area = (x2_g - x1_g + 1) * (y2_g - y1_g + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def analyze_results(results, ground_truth, image_name):
    iou_scores = []

    for result in results.xyxy[0]:
        pred_box = result[:4].int().tolist()
        pred_cls = int(result[5])
        pred_conf = result[4].item()

        if image_name in ground_truth:
            gt_records = ground_truth[image_name]
            for gt in gt_records:
                gt_box = [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
                gt_cls = gt['cls_id']

                if pred_cls == gt_cls:
                    iou = calculate_iou(pred_box, gt_box)
                    iou_scores.append(iou)

    return iou_scores

def plot_distribution(iou_scores):
    fig, ax = plt.subplots()
    ax.hist(iou_scores, bins=20, alpha=0.75, color='blue', edgecolor='black')
    ax.set_title('IoU Score Distribution')
    ax.set_xlabel('IoU Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def image_input(src, model, ground_truth):
    if src == 'Upload your own Image':
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
            st.image(img, caption='Uploaded Image', use_column_width=False, width=300)
            img_tensor = process_image(img)
            if img_tensor is not None:
                try:
                    st.write("Making prediction...")
                    results = model(img_tensor)[0]  # Corrected prediction call
                    img_with_bboxes = draw_bboxes(img, results, ground_truth, image_name)
                    st.image(img_with_bboxes, caption='Predicted Heart Segmentation', use_column_width=False, width=300)
                    iou_scores = analyze_results(results, ground_truth, image_name)
                    plot_distribution(iou_scores)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    elif src == 'From sample Images':
        selected_image = st.sidebar.slider("Select random image from test set.", 1, 50)
        image_name = f"{selected_image}.jpg"
        image_url = f"https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/{image_name}"
        try:
            st.write("Downloading sample image from URL...")
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption='Sample Image', use_column_width=False, width=300)
            img_tensor = process_image(image)
            if img_tensor is not None:
                try:
                    st.write("Making prediction...")
                    results = model(img_tensor)[0]  # Corrected prediction call
                    img_with_bboxes = draw_bboxes(image, results, ground_truth, image_name)
                    st.image(img_with_bboxes, caption='Predicted Heart Segmentation', use_column_width=False, width=300)
                    iou_scores = analyze_results(results, ground_truth, image_name)
                    plot_distribution(iou_scores)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        except Exception as e:
            st.error(f"Error downloading sample image: {e}")

def main():
    gif_url = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/blob/main/WholeHeartSegment_ErrorMap_WhiteBg.gif?raw=true"
    gif_path = "WholeHeartSegment_ErrorMap_WhiteBg.gif”
    if not os.path.exists(gif_path):
    try:
        st.write("Downloading GIF from URL...")
        response = requests.get(gif_url)
        with open(gif_path, 'wb') as f:
            f.write(response.content)
        st.write("GIF downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading gif: {e}")

if os.path.exists(gif_path):
    try:
        st.image(gif_path, width=500)
    except Exception as e:
        st.error(f"Error displaying image: {e}")
else:
    st.error(f"Error opening '{gif_path}'. File not found.")

st.title("3D Heart MRI Image Segmentation")
st.subheader("AI driven apps made by Md Abu Sufian")
st.header("👈🏽 Select the Image Source options")
st.sidebar.title('⚙️Options')

src = st.sidebar.radio("Select input source.", ['From sample Images', 'Upload your own Image'])

model = load_model()
ground_truth = load_ground_truth()

if model is not None and ground_truth is not None:
    image_input(src, model, ground_truth)
if name == ‘main’:
main()
