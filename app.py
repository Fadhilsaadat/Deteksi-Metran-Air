import streamlit as st
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = 'best.pt'

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Deteksi Angka Meteran Air")
st.caption('Upload a photo with this :blue[hand signals]: :+1:, :hand:, :i_love_you_hand_sign:, and :spock-hand:.')
example_image = "test.jpg"
st.image(example_image, caption="Contoh Deteksi Meteran Air", width=300)
st.caption('Then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Inisialisasi model EasyOCR
reader = easyocr.Reader(['en'])

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

res_plotted = None  # Inisialisasi variabel res_plotted di luar cakupan

if st.sidebar.button('Detect Objects and Text'):
    if source_img is not None:
        uploaded_image = Image.open(source_img)  # Mendefinisikan uploaded_image
        res = model.predict(uploaded_image, conf=confidence, iou=0.5)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]

        with col2:
            st.image(res_plotted, caption='Detected Image', use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("No image is uploaded yet.")

        # Ekstrak teks dari gambar hasil deteksi objek
        if res_plotted is not None:
            detected_text = reader.readtext(np.array(res_plotted))
            with st.expander("Detected Text"):
                for detection in detected_text:
                    st.write(f"Text: {detection[1]}, Confidence: {detection[2]}")
        else:
            st.warning("You need to run object detection first.")
    else:
        st.warning("Please upload an image before running object detection and text extraction.")





