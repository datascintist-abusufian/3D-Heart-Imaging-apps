import streamlit as st
import torch
from PIL import Image
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from torchvision.transforms import transforms
from io import BytesIO

# Function to download and verify the model
def download_model():
    st.write("Downloading model...")
    model_path = "models/yolov5s.pt"
    url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'

    # Known file size of yolov5s.pt
    expected_size = 14602242

    if not os.path.exists(model_path) or os.path.getsize(model_path) != expected_size:
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        try:
            with requests.get(url, stream=True, verify=False) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            st.write("Model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error during model download: {e}")
            return None

    # Verify the file size
    if os.path.getsize(model_path) != expected_size:
        st.error("Downloaded model file size is incorrect. The file may be corrupted.")
        os.remove(model_path)
        return None

    return model_path

@st.cache_resource
def load_model():
    model_path = download_model()
    if model_path is None:
        return None
    
    try:
        st.write("Loading model from path...")
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    return model

def process_image(image):
    st.write("Processing image...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    try:
        image = transform(image).unsqueeze(0)
        st.write("Image processed successfully.")
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def image_input(src, model):
    if src == 'Upload your own Image':
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            img_tensor = process_image(img)
            if img_tensor is not None:
                try:
                    st.write("Making prediction...")
                    with torch.no_grad():
                        results = model(img_tensor)
                        results.render()
                        for img in results.ims:
                            img_pil = Image.fromarray(img)
                            st.image(img_pil, caption='Predicted Heart Segmentation')
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    elif src == 'From sample Images':
        image_url = "https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/1.jpg"
        try:
            st.write("Downloading sample image from URL...")
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Sample Image', use_column_width=True)
            img_tensor = process_image(image)
            if img_tensor is not None:
                try:
                    st.write("Making prediction...")
                    with torch.no_grad():
                        results = model(img_tensor)
                        results.render()
                        for img in results.ims:
                            img_pil = Image.fromarray(img)
                            st.image(img_pil, caption='Predicted Heart Segmentation')
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
    
    st.title("3D Heart MRI Image Segmentation")
    st.subheader("AI driven apps made by Md Abu Sufian")
    st.header("👈🏽 Select the Image Source options")
    st.sidebar.title('⚙️Options')
    src = st.sidebar.radio("Select input source.", ['From sample Images', 'Upload your own Image'])

    model = load_model()
    if model is not None:
        image_input(src, model)

if __name__ == '__main__':
    main()
