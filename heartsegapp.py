
import streamlit as st
import torch
from PIL import Image
import glob
from datetime import datetime
import os
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import time

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
    return model

model = load_model()

def imageInput(src):
    if src == 'Upload your own Image':
        image_file = st.file_uploader("Upload a 3D Heart MRI Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded 3D Heart MRI Image', use_column_width=True)
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            try:
                # Call Model prediction
                pred = model(imgpath)
                pred.render()  # render bbox in image
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
        imgpath = glob.glob('data/images/test/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1)
        image_file = imgpath[imgsel - 1]
        submit = st.button("Predict the Heart Segmentation")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:
            if image_file is not None and submit:
                try:
                    pred = model(image_file)
                    pred.render()  # render bbox in image
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
