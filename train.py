import pickle
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

feature_file = open('features.out', 'rb')
label_file = open('labels.out', 'rb')

X = pickle.load(feature_file)
y = pickle.load(label_file)


model = Sequential()
model.add(Conv2D(64, (10, 10), 2, input_shape = (100, 100, 3))) #Convolution layer
model.add(Activation('relu')) #Activation for convolutions
model.add(MaxPooling2D(pool_size = (2, 2))) #Pooling convolutions

model.add(Conv2D(64, (10, 10)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten()) #Flattening 2D layer to 1D
model.add(Dense(64))

model.add(Dense(len(y[0])))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, epochs = 2, batch_size = 10)

model.save('FruitTrainCNN.model')
