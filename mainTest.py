import cv2
from keras.models import load_model

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('.\pred\pred56.jpg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
img = Image.fromarray(image, 'RGB')
img = img.resize((64,64))
imgplot = plt.imshow(image)
plt.show()

import numpy as np
img = np.array(img)
print(img)
print(img.shape)
input_img = np.expand_dims(img, axis=0)
print(input_img)
print(input_img.shape)

result=model.predict(input_img)
print(result)