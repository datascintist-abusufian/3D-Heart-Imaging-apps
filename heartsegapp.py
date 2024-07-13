import streamlit as st
import torch
from PIL import Image
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from torchvision.transforms import transforms
from io import BytesIO

# Cache the model loading
@st.cache_resource
def load_model():
    st.write("Loading model...")
    model_path = "models/yolov5s.pt"

    # Check if model exists
    if not os.path.exists(model_path):
        # Disable SSL warnings
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        
        url = 'https://github.com/datascintist-abusufian/3D-Heart-Imaging-apps/blob/main/models/yolov5s.pt?raw=true'
        
        # Download the model
        try:
            st.write("Downloading model from URL...")
            response = requests.get(url, stream=True, verify=False)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write("Model downloaded successfully.")
            else:
                st.error("Failed to download the model")
                return None
                 
        except Exception as e:
            st.error(f"Error during model download: {e}")
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

# Process the image to a tensor
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

# Function to handle image input
def image_input(src, model):
    if src == 'Upload your own Image':
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_tensor = process_image(img)
            if img_tensor is not None:
                try:
                    st.write("Making prediction...")
                    with torch.no_grad():
                        pred = model(img_tensor)
                        pred.render()
                        for im in pred.ims:
                            im_base64 = Image.fromarray(im)
                            st.image(im_base64, caption='Predicted Heart Segmentation')
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    elif src == 'From sample Images':
        image_url = "https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/1.jpg"
        try:
            st.write("Downloading sample image from URL...")
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            img_tensor = process_image(image)
            if img_tensor is not None:
                try:
                    st.write("Making prediction...")
                    with torch.no_grad():
                        pred = model(img_tensor)
                        pred.render()
                        for im in pred.ims:
                            im_base64 = Image.fromarray(im)
                            st.image(im_base64, caption='Predicted Heart Segmentation')
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        except Exception as e:
            st.error(f"Error downloading sample image: {e}")

# Main function
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
