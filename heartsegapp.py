import streamlit as st
import torch
from PIL import Image
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import time
from torchvision.transforms import transforms

# Cache the model loading
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    model_path = "models/yoloTrained.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Disable SSL warnings
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        
        url = 'https://archive.org/download/yoloTrained/yoloTrained.pt'
        
        start_dl = time.time()  # Start the timer
        
        # Download the model
        try:
            response = requests.get(url, stream=True, verify=False)  # Note: verify=False bypasses SSL certificate verification
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Failed to download the model")
                return None
                
        except Exception as e:
            st.error(f"Error during model download: {e}")
            return None

        finished_dl = time.time()  # End the timer
        st.write(f"Model Downloaded in {finished_dl-start_dl:.2f} seconds")
    
    try:
        model = torch.load(model_path)
        model.eval()  # Set the model to evaluation mode
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    st.write(f"Model type: {type(model)}")  # Print model type to Streamlit
    return model

# Process the image to a tensor
def process_image(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    try:
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to handle image input
def imageInput(src):
    if src == 'Upload your own Image':
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_file is not None:
            with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())         
            try:
                img_tensor = process_image(os.path.join("tempDir",uploaded_file.name))
                if img_tensor is not None:
                    pred = model(img_tensor)
                    # Add your prediction handling code here
            except Exception as e:
                st.error(f"Error during prediction: {e}")
 elif src == 'From sample Images':
        # List of URLs to your GitHub-hosted images (raw version)
        base_url = "https://raw.githubusercontent.com/datascintist-abusufian/3D-Heart-Imaging-apps/main/data/images/test/"
        github_image_urls = [base_url + f"{i}.jpg" for i in range(1, 51)]  # For images 1.jpg to 50.jpg

        selected_url = st.selectbox("Select an image:", github_image_urls)
        if selected_url:
            try:
                response = requests.get(selected_url)
                image = Image.open(BytesIO(response.content))
                img_tensor = process_image(image)
                if img_tensor is not None:
                    with torch.no_grad():
                        pred = model(img_tensor)

                    # Assuming your model has a render function
                    pred.render()
                    for im in pred.ims:
                        im_base64 = Image.fromarray(im)
                        # Handling for display in Streamlit
                        st.image(im_base64, caption='Predicted Heart Segmentation')
            except Exception as e:
                st.error(f"Error during prediction: {e}")


# Main function
def main():
    st.image("logo.jpg", width=500)
    st.title("3D Heart MRI Image Segmentation")
    st.header("👈🏽 Select the Image Source options")
    st.sidebar.title('⚙️Options')
    src = st.sidebar.radio("Select input source.", ['From sample Images', 'Upload your own Image'])

    global model
    model = load_model()
    if model is not None:
        imageInput(src)

if __name__ == '__main__':
    main()
