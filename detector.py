import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from IPython.display import display
import glob
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils


my_model = load_model('model/model_cnn.h5')
prototxt_path = "detector/deploy.prototxt"
weight_path = "detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_path, weight_path)

def detectPredictMask(frame, faceNet, maskNet):
    print('masuk detectPredictMask')
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Compute the (x, y)-coordinates of the bounding box for the object
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # Ensure the bounding boxes fall within the dimensions of the frame
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    # Extract the face ROI, convert it from BGR to RGB channel
    # Ordering, resize it to 224x224, and preprocess it
    face = frame[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    img = np.array(face, dtype='float')
    img = img.reshape(1, 224, 224, 3)
    
    # Predict the frame
    preds = maskNet.predict(img)

    # Return a 2-tuple of the face locations and their corresponding locations
    return ((startX, startY, endX, endY), preds[0][0])

def framing(img):
    print('masuk framing')
    frame = cv2.imread(img)
    frame = imutils.resize(frame)
    
    (locs, preds) = detectPredictMask(frame, faceNet, my_model)
    
    # Unpack the bounding box and predictions
    (startX, startY, endX, endY) = locs
    result = preds
        
    # Determine the class label and color we'll use to draw the bounding box and text
    color = (0, 255, 0)
    status =" Wearing Mask"
    if (result == 1):
        status =" Not Wearing Mask"
        color = (0, 0, 255)

    font = cv2.FONT_HERSHEY_DUPLEX

    stroke = 1
    cv2.putText(frame, status, (startX, startY - 10), font, 0.5, color, stroke, cv2.LINE_AA)

    stroke = 2
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, stroke)
        
    # Showing frame
    # cv2.imshow('DETECTING', frame)
    return frame