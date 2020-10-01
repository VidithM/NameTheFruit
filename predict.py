import cv2, pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as mdl

IMG_PATH = './fruits-360/mango.jpg' #Put your image to predict here

cat_file = open('categories.out', 'rb')
categories = pickle.load(cat_file)

model = mdl.load_model('FruitTrainCNN.model')

img = cv2.imread(IMG_PATH)
img = cv2.resize(img, (100, 100))

img = np.array(img).reshape(-1, 100, 100, 3)
prediction = model.predict(img)

max_val = -1
max_cat = 0
for cat in range(len(prediction[0])):
    if(prediction[0][cat] > max_val):
        max_val = prediction[0][cat]
        max_cat = cat

print(categories[max_cat])


