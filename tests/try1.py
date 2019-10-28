# -*- coding: utf-8 -*-
"""
@ COMP 551: Applied Machine Learning (Winter 2019) 
@ Mini-project 3: Kaggle Modified MNIST classification
# Team Members: 

@ Hair Albeiro Parra Barrera 
@ ID: 260738619 

@ Tommy Luo
@ ID: ???
    
@ Logan Ralston
@ ID: 
"""

import os
import cv2 
import time
import random 
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split 


import tensorflow as tf 
from tensorflow.keras.utils import normalize # do we actually need this? 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard 
from tensorflow.keras.preprocessing import image


# GPU tensorflow GPU configuration 
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# read the train and test images  
train_images = pd.read_pickle('../data_raw/train_max_x') 
test_images = pd.read_pickle('../data_raw/test_max_x')
train_labels = pd.read_csv("../data_raw/train_max_y.csv").iloc[:,1]
CATEGORIES = sorted(set(train_labels))

# split into a further validation set 
train_labeled_img = train_images.reshape(50000, 128,128)
plt.imshow(train_images[100])


# further split into training and testing data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(train_images, train_labels, 
                                                                    test_size=.01, 
                                                                    random_state=42) 

# Get unique name for the model 
NAME = "Modified_MNIST_keras_cnn_64x2_{}".format(int(time.time()))

# tensorboard callbacks 
tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))


# reshape to tensorflow format 
X_train_new = np.reshape(X_train_new, (X_train_new.shape[0], X_train_new.shape[1], X_train_new.shape[2], 1))
X_test_new = np.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], X_test_new.shape[2], 1))

# normalize by max num_pixels
X_train_new = X_train_new / 255.0 
X_test_new = X_test_new / 255.0 


## Build the model ###

# Building the model uses the following pattern: 
# model.add(layer_type, window, input_shape) 


model = Sequential() 

# Convolutional later 
model.add(Conv2D(64, (3,3), input_shape = (128,128,1) ) ) # Convolutional NN layer  
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) 

# Convolutional later 
model.add(Conv2D(64, (3,3)))  # Convolutional NN layer  
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) 

# Flatten Layer 2D -> 1D 
model.add(Flatten()) 

# Output layer 
model.add(Dense(10)) 
model.add(Activation('softmax')) 

# Model compilation 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# display the architecture summary 
model.summary()


### Fitting the model ### 

BATCH_SIZE = 32
EPOCHS = 5 

# fit the model
model.fit(X_train_new, y_train_new, batch_size=BATCH_SIZE, 
          epochs = EPOCHS, 
          callbacks=[tensorboard]) 

