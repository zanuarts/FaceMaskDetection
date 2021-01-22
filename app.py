from flask import Flask, request, render_template
from PIL import Image
from tensorflow.keras.models import load_model
import os
from preprocessing import preprocessing_image, get_encoding
from detector import detectPredictMask
import imutils
import cv2

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/img/uploaded/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])

model = load_model('model/model_cnn.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    
    if request.method == 'GET':
        return render_template('index.html', value='hi')
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs('static/img/uploaded')

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename.replace(' ','_')
            dest = UPLOAD_FOLDER+filename
            file.stream.seek(0)
            file.save(dest)
            file.stream.seek(0)
            image = dest
            image = preprocessing_image(image)
            encoded_image = get_encoding(model, image)
            # img = cv2.imread(image)
            # detect = detectPredictMask(image)
            return render_template('result.html', result=encoded_image.upper(), image_file=dest)

if __name__ == '__main__':
    app.run(debug=True)