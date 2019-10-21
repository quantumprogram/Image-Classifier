# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:04:35 2019

@author: Souptik
"""

from keras.models import load_model
from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
import numpy as np

#array to convert the digits given out my model to number
values=['buildings','forest','glacier','mountain','sea','street']

classifier = load_model('imagingclass.h5')

classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

image1 = load_img('seg_pred/seg_pred/1585.jpg', target_size=(128,128))    #change the image directory here to add image 
image1 = np.reshape(image1,[1,128,128,3])
image1=image1/255
classes = classifier.predict_classes(image1)

print(values[classes[0]])