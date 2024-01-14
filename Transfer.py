import cv2
import os
from PIL import Image
import numpy as np

image_directory='./datasets/'  #This line sets the path to the directory containing the dataset of brain tumor images.

no_tumor_images = os.listdir(image_directory+'no/')     #lists all images in no directory 
yes_tumor_images = os.listdir(image_directory+'yes/')   #s list all the image files in the "no" and "yes" subdirectories
dataset = []
label = []

# print(no_tumor_images)

path = 'no0.jpg'

# print(path.split('.'))      #['no0', 'jpg']
path.split('.')[1]

for i, image_name in enumerate(no_tumor_images): #This loop iterates through the filenames in the "no" subdirectory. It loads each image using OpenCV, converts it to an RGB format using PIL, resizes it to 64x64 pixels, and appends the image data to the dataset list. It also appends the label 0 (indicating "no tumor") to the label list.
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
        
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB') # converting opencv format into Python imaging library(PIL) format
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
        
        
print(len(label))
print(len(dataset))

#converting dataset into numpy array:

#PREPROCESSING STARTS
dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split  #Split arrays or matrices into random train and test subsets.

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state= 0) #20% data will be used for testing

import tensorflow as tf  #deployment of machine learning models, particularly neural networks, for a wide range of tasks, from image and speech recognition to natural language processing and reinforcement learning.
from tensorflow import keras
from tensorflow.keras.utils import normalize

# x_test = x_test/255
x_train = normalize(x_train, axis = 1)
x_test = normalize(x_test, axis = 1)
# x_train = x_train/255

from keras.applications.vgg16 import VGG16  #VGG16 is a convolutional neural network (CNN) architecture, known for its simplicity and effectiveness in image classification tasks.
from keras.layers import Input, Dense, Flatten
from keras.models import Model

vgg = VGG16(input_shape = (64,64,3),weights = 'imagenet',include_top = False) #This part imports the VGG16 model from Keras with pre-trained weights (ImageNet) and excludes the top layer (the fully connected layers) because you'll add your own classification layers later.


#transfer learning
#This loop sets all layers in the VGG16 model to non-trainable. This means that the pre-trained weights will not be updated during training
# In many cases, you want to use a pre-trained deep learning model like VGG as a feature extractor. By freezing the layers, you ensure that the model retains the features it has learned from a large dataset (e.g., ImageNet).
for layer in vgg.layers:
    layer.trainable = False
    
x = Flatten()(vgg.output)
prediction = Dense(1,activation = 'softmax')(x)     #Dense layer with one unit and softmax activation is added for binary classification (0 or 1).
model = Model(inputs = vgg.input,outputs = prediction)

model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.0001), metrics = ['accuracy'])

#validation
model.fit(x_train, y_train,
          batch_size = 16,         # is a hyperparameter that determines the number of samples used in each iteration of training when you call the model.fit method
          epochs = 2,   #The epochs parameter determines how many times the model will go through the entire dataset during training.
          validation_data = (x_test, y_test), # optional parameter. If provided, it should be a tuple (x_val, y_val) containing validation data
          shuffle = False)

model.save('BrainTumor10EpochsTransferL.h5') 