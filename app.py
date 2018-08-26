import flask
from flask import Flask, request, jsonify
from werkzeug import secure_filename
from keras import models
from keras import backend as K
import cv2
import numpy as np
import pandas as pd
import pathlib

"""
Import all the dependencies you need to load the model, 
preprocess your request and postprocess your result
"""
app = Flask(__name__)
# MODEL_PATH = pathlib.Path('trained_model')
MODEL_PATH = pathlib.Path('..')
ALLOWED_EXTENSIONS = ['jpg', 'png']

def load_model(MODEL_PATH):
    """Load the model"""
    K.clear_session()
    model = models.load_model(MODEL_PATH/'cats_and_dogs_10.h5')
    print("Model loaded")
    return model

def data_preprocessing(data):
    """Preprocess the request data to transform it to a format that the model understands"""
    print("Preprocessing")
    # CV2
    nparr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (150, 150))
    img = img[None, ...]
    print("Image shape", img.shape)
    return img

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Every incoming POST request will run the `evaluate` method
# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/', methods=["POST"])
def evaluate():
    """Preprocessing the data and evaluate the model"""
    model = load_model(MODEL_PATH)
    if flask.request.method == "POST":
        input_file = request.files.get('file')
        print("Received file")
        if not input_file:
            return BadRequest("File not present in request")

        filename = secure_filename(input_file.filename)
        if filename == '':
            return BadRequest("File name is not present in request")
        if not allowed_file(filename):
            return BadRequest("Invalid file type")

        input_file = input_file.read()
        img = data_preprocessing(input_file)

        pred = model.predict(img)

        return jsonify({"Prediction": pred.tolist()})


# Load the model and run the server
if __name__ == "__main__":
    print(("* Loading model and starting Flask server..."
        "please wait until server has fully started"))
    load_model(MODEL_PATH)
    app.debug = True
    app.run(host='0.0.0.0')
