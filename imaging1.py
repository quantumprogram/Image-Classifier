# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:28:23 2019

@author: Souptik
"""
""" The documentation for all the functions used below are availabe in https://keras.io/ I will recommed
checking it out""" 



from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#initialized the neural network
classifier = Sequential()    
#added the convolution layer with 64 features each of size 3X3 and the image is of size 128X128X3(colour image)
#activation function is rectified linear unit 
classifier.add(Convolution2D(64, 3, 3, input_shape=(128, 128, 3), activation='relu')) 
#used pooling on feature map to reduce the size to half so 2 
classifier.add(MaxPooling2D(pool_size=(2,2)))  
#the first convolution was not complex enough to predict so another layer
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
#pooling on it again
classifier.add(MaxPooling2D(pool_size=(2,2))) 
#flattened the values to apply to a ann 
classifier.add(Flatten())
#created the connections of ann with 128 nodes in hidden layers and 6 on output layer
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 6, activation='sigmoid'))
#compiled the model
classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
#called image data generator to generate some more data by changing zoom,shear etc and rescaled the values
#to be in 0 to 1
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#rescaled the test data
test_datagen = ImageDataGenerator(rescale=1./255)
#took the data from training folder
train_generator = train_datagen.flow_from_directory(
        'seg_train/seg_train',                  #change this value to your training data directory
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'seg_test/seg_test',                    #change this value to your test data directory
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical')
import tensorflow as tf

#used tenserflow to fit the model using gpu
with tf.device('gpu'):                          #remove this line if you do not have a gpu and using just cpu
    classifier.fit_generator(
        train_generator,
        steps_per_epoch=14000,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=3000)
#saved the model to hardrive    
classifier.save('imagingclass.h5')  

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
#loaded the model 
classifier = load_model('imagingclass.h5')

classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])


#loaded a image to check whether it predicts the right class for it
image1 = load_img('seg_pred/seg_pred/4472.jpg', target_size=(128,128))
image1 = np.reshape(image1,[1,128,128,3])
image1=image1/255
classes = classifier.predict_classes(image1)

print(classes)