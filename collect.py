import cv2, os, random, pickle
import numpy as np

training_dir = './fruits-360/Training'
train_data = []
categories = []

numdir = 10

for dir_name in os.listdir(training_dir):
    label = dir_name.split(' ')[0]
    if(not label in categories):
        categories.append(label)

for dir_name in os.listdir(training_dir):
    img_dir = training_dir + '/' + dir_name
    label = dir_name.split(' ')[0]

    for fl in os.listdir(img_dir):
        img = cv2.imread(img_dir + '/' + fl)
        img = cv2.resize(img, (100, 100))
        
        actual = np.zeros(len(categories))
        actual[categories.index(label)] = 1
        train_data.append((img, actual))


random.shuffle(train_data)
X = []
y = []

for img, label in train_data:
    X.append(img)
    y.append(label)

X = np.array(X).reshape(-1, 100, 100, 3)
y = np.array(y).reshape(-1, len(categories))
print(y[0])

feature_file = open('features.out', 'wb')
label_file = open('labels.out', 'wb')
cat_file = open('categories.out', 'wb')

pickle.dump(X, feature_file)
pickle.dump(y, label_file)
pickle.dump(categories, cat_file)

feature_file.close()
label_file.close()
cat_file.close()

