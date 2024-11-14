import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests
from torchvision.transforms import transforms
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score

# --- Configuration ---
MODEL_URL = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/raw/main/yolov5s.pt"
MODEL_PATH = "yolov5s.pt"
GIF_PATH = "WholeHeartSegment_ErrorMap_WhiteBg.gif"
SAMPLE_IMAGES_DIR = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/tree/main/data/images/test"
CLASS_NAMES = {0: 'left ventricle', 1: 'right ventricle'}
THRESHOLD = 0.3  # Lowered confidence threshold for considering a detection as correct

@st.cache_resource
def load_model():
    model_file = MODEL_PATH
    if not os.path.exists(model_file):
        st.write("Downloading model...")
        response = requests.get(MODEL_URL)
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

def process_image(image):
    st.write("Processing image...")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Ensure this matches your training configuration
        transforms.ToTensor(),          # Ensure this matches your training configuration
    ])

    try:
        image = transform(image).unsqueeze(0)
        st.write("Image processed successfully.")
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def draw_bboxes_and_masks(image, results, ground_truth_mask=None):
    img = np.array(image)
    class_names = {0: 'left ventricle', 1: 'right ventricle'}
    confidence_scores = []

    # Ensure predicted mask dimensions align with the ground truth mask dimensions
    img_height, img_width = img.shape[:2]
    pred_mask = np.zeros((img_height, img_width), dtype=np.uint8)  # Adjust size to match input image

    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            label = class_names.get(cls_id, 'Unknown')
            score = conf

            confidence_scores.append(score)

            if score > THRESHOLD:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} {score:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

                cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (255, 0, 0), -1)
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw segmentation mask
                if results.masks is not None and len(results.masks) > i:
                    mask = results.masks[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))  # Resize mask to fit bounding box
                    mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                    pred_mask[y1:y2, x1:x2] = mask_resized
                    roi = img[y1:y2, x1:x2]
                    roi[np.where(mask_resized == 255)] = (0, 255, 0)  # Overlay the mask with green color
                else:
                    st.write(f"No mask found for bounding box {i} with confidence {score}")

    return img, confidence_scores, pred_mask

def calculate_metrics(true_mask, pred_mask):
    img_height, img_width = pred_mask.shape[:2]
    true_mask_resized = cv2.resize(true_mask, (img_width, img_height))  # Resize ground truth to match pred_mask

    # Debug statements to check mask contents
    st.write(f"True mask unique values: {np.unique(true_mask_resized)}")
    st.write(f"Predicted mask unique values: {np.unique(pred_mask)}")

    intersection = np.logical_and(true_mask_resized, pred_mask)
    union = np.logical_or(true_mask_resized, pred_mask)
    dice = 2 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(true_mask_resized))
    iou = np.sum(intersection) / np.sum(union)

    precision = precision_score(true_mask_resized.flatten(), pred_mask.flatten(), average='binary')
    recall = recall_score(true_mask_resized.flatten(), pred_mask.flatten(), average='binary')

    return dice, iou, precision, recall

def plot_distribution(confidence_scores):
    fig, ax = plt.subplots(figsize=(5, 3))  # Set figure size here
    ax.hist(confidence_scores, bins=10, alpha=0.75, color='blue', edgecolor='black')
    ax.set_title('Confidence Score Distribution')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def image_input(src, model):
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

                    # Load ground truth mask for comparison (replace with actual loading mechanism)
                    ground_truth_mask = np.zeros((640, 640), dtype=np.uint8)  # Replace with actual ground truth mask
                    img_with_bboxes, confidence_scores, pred_mask = draw_bboxes_and_masks(img, results, ground_truth_mask)

                    st.image(img_with_bboxes, caption='Predicted Heart Segmentation', use_column_width=False, width=300)
                    plot_distribution(confidence_scores)

                    if np.any(pred_mask):  # Only calculate metrics if there is a predicted mask
                        dice, iou, precision, recall = calculate_metrics(ground_truth_mask, pred_mask)
                        st.write(f"Dice Coefficient: {dice:.2f}")
                        st.write(f"IoU: {iou:.2f}")
                        st.write(f"Precision: {precision:.2f}")
                        st.write(f"Recall: {recall:.2f}")
                    else:
                        st.write("No masks found in the prediction.")
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

                    # Load ground truth mask for comparison (replace with actual loading mechanism)
                    ground_truth_mask = np.zeros((640, 640), dtype=np.uint8)  # Replace with actual ground truth mask
                    img_with_bboxes, confidence_scores, pred_mask = draw_bboxes_and_masks(image, results, ground_truth_mask)

                    st.image(img_with_bboxes, caption='Predicted Heart Segmentation', use_column_width=False, width=300)
                    plot_distribution(confidence_scores)

                    if np.any(pred_mask):  # Only calculate metrics if there is a predicted mask
                        dice, iou, precision, recall = calculate_metrics(ground_truth_mask, pred_mask)
                        st.write(f"Dice Coefficient: {dice:.2f}")
                        st.write(f"IoU: {iou:.2f}")
                        st.write(f"Precision: {precision:.2f}")
                        st.write(f"Recall: {recall:.2f}")
                    else:
                        st.write("No masks found in the prediction.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        except Exception as e:
            st.error(f"Error downloading sample image: {e}")

def main():
    gif_url = "https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/blob/main/WholeHeartSegment_ErrorMap_WhiteBg.gif?raw=true"
    gif_path = "WholeHeartSegment_ErrorMap_WhiteBg.gif"

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

    st.title("3D MRI Heart Imaging")
    st.subheader("AI driven apps made by Md Abu Sufian")
    st.header("👈🏽 Select the Image Source options")
    st.sidebar.title('⚙️Options')

    src = st.sidebar.radio("Select input source.", ['From sample Images', 'Upload your own Image'])

    model = load_model()

    if model is not None:
        image_input(src, model)

if __name__ == '__main__':
    main()
