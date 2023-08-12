
import streamlit as st
import torch
from PIL import Image
import glob
from datetime import datetime
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import time
from torchvision.transforms import transforms

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    model_path = "models/yoloTrained.pt"
    
    if not os.path.exists(model_path):
        # Disable SSL warnings
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        
        url = 'https://archive.org/download/yoloTrained/yoloTrained.pt'
        
        start_dl = time.time()  # Start the timer
        
        # Download the model
        response = requests.get(url, stream=True, verify=False)  # Note: verify=False bypasses SSL certificate verification
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        finished_dl = time.time()  # End the timer
        st.write(f"Model Downloaded in {finished_dl-start_dl:.2f} seconds")
    
    model = torch.load(model_path)
    st.write(f"Model type: {type(model)}")  # Print model type to Streamlit
    return model

model = load_model()

def process_image(img_path):
    # This is a simple function to process an image to tensor.
    # You might need to adjust it based on your model requirements.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)
    return image

def imageInput(src):
    if src == 'Upload your own Image':
        # ... (rest of your code)

            try:
                # Process the image to tensor
                img_tensor = process_image(imgpath)

                # Make prediction
                with torch.no_grad():
                    pred = model(img_tensor)

                # Since you mentioned 'render' and 'ims', I'm assuming 
                # you're working with a modified YOLO or similar object detection model
                pred.render()
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)

                # Display prediction
                img_ = Image.open(outputpath)
                with col2:
                    st.image(img_, caption='Predicted Heart Segmentation', use_column_width=True)
            except Exception as e:
                st.write(f"Error during prediction: {e}")

    elif src == 'From sample Images':
        # ... (rest of your code)

                try:
                    # Process the image to tensor
                    img_tensor = process_image(image_file)

                    # Make prediction
                    with torch.no_grad():
                        pred = model(img_tensor)

                    pred.render()
                    for im in pred.ims:
                        im_base64 = Image.fromarray(im)
                        im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Predicted Heart Segmentation')
                except Exception as e:
                    st.write(f"Error during prediction: {e}")

def main():
    st.image("logo.jpg", width=500)
    st.title("3D Heart MRI Image Segmentation")
    st.header("👈🏽 Select the Image Source options")
    st.sidebar.title('⚙️Options')
    src = st.sidebar.radio("Select input source.", ['From sample Images', 'Upload your own Image'])
    imageInput(src)

if __name__ == '__main__':
    main()
my_dict = {"key": "value"}
print(my_dict["key"])
