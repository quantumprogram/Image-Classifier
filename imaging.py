# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:22:08 2019

@author: Souptik
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential() 
classifier.add(Convolution2D(64, 3, 3, input_shape=(128, 128, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))  
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))  
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 6, activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'seg_train/seg_train',
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'seg_test/seg_test',
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical')
import tensorflow as tf
with tf.device('gpu'):
    classifier.fit_generator(
        train_generator,
        steps_per_epoch=14000,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=3000)
    
classifier.save('imagingclass.h5')  

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

classifier = load_model('imagingclass.h5')

classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

image1 = load_img('seg_pred/seg_pred/4472.jpg', target_size=(128,128))
image1 = np.reshape(image1,[1,128,128,3])
image1=image1/255
classes = classifier.predict_classes(image1)

print(classes)