from flask import Flask, Response, request
from flask import send_from_directory as send
from flask import render_template as render

import os, shutil
import numpy as np
import cv2, pickle
import tensorflow.keras.models as mdl

UPLOAD_PATH = './uploads'

app = Flask(__name__, template_folder = './static/templates/')

#Runs the uploaded image through the CNN, returns classification result as string
def processUpload(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (100, 100))
    img = np.array(img).reshape(-1, 100, 100, 3)

    cat_file = open('../categories.out', 'rb')
    categories = pickle.load(cat_file)
    model = mdl.load_model('../FruitTrainCNN.model')

    prediction = model.predict(img)
    max_val = -1
    max_cat = 0
    for cat in range(len(prediction[0])):
        if(prediction[0][cat] > max_val):
            max_val = prediction[0][cat]
            max_cat = cat
    return categories[max_cat]

#Removes all previous uploads (we don't need them)
def clearUploads():
    for root, dirs, files in os.walk(UPLOAD_PATH):
        for file in files:
            os.remove(os.path.join(root, file))

@app.route('/', methods = ['POST', 'GET'])
def home():
    resp = {'alert' : ''}
    if(request.method == 'POST'):
        upload = request.files['data-in']
        ftype = upload.content_type
        if(not(ftype.split('/')[0] == 'image')):
            resp['alert'] = 'Please upload an image to classify'
        else:
            path = os.path.join(UPLOAD_PATH, upload.filename)
            clearUploads()
            upload.save(path)
            resp['result'] = processUpload(path)
    return render('index.html', **resp)

if __name__ == '__main__':
    app.run()