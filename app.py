import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def mask_image():
    global RGB_img
    print("Waiting....")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector","res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    print("Loading, Have Patience...")
    model = load_model("mask_detector.model")

    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
    print("Computing....")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence_prob = detections[0, 0, i, 2]
    # Filtering the weak prob
        if confidence_prob > 0.5:
            # bounding box calculation
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # Extracting the Region of Interest
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask Image" if mask > withoutMask else "No Mask Image"
            color = (0, 255, 0) if label == "Mask Image" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_image()

def mask_detection():
    local_css("css\styles.css")
    st.markdown('<h1 align="center"> Face Mask Detector </h1>', unsafe_allow_html=True)
    activities = ["Image File"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.markdown('<h2 align="center">Detects whether a person is wearing mask or not</h2>', unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.markdown("Kindly Upload your Image")
    image_file = st.file_uploader("", type=['jpg'])  # upload image
    if image_file is not None:
        our_image = Image.open(image_file)  # making compatible to PIL
        im = our_image.save('./images/out.jpg')
        saved_image = st.image(image_file, caption='', use_column_width=True)
        st.markdown('<h3 align="center">Success ! Image has been properly loaded</h3>', unsafe_allow_html=True)
        if st.button('Process'):
            st.image(RGB_img, use_column_width=True)
    
mask_detection()
