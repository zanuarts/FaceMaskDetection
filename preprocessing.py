import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, RepeatVector, Embedding
# from inference import 

def preprocessing_image(img_path):
    im = image.load_img(img_path, target_size=(224, 224, 3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis = 0)
    im = preprocess_input(im)
    return im

def get_encoding(model, img):
    img = np.array(img, dtype='float')
    img = img.reshape(1, 224, 224, 3)
    pred = model.predict(img)
    idx = pred[0][0]
    if (idx):
        print("Not Wearing Masker")
    else:
        print("Wearing Masker")
    return pred