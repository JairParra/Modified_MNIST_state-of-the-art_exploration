

import tensorflow as tf 

tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


# *****************************************************************************


### 2. Deep Learning with Python, Tensorflow, and Keras ###

# We wil use the Kaggle Cats and Dogs Dataset with a NN

import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2 

DATADIR = "C:\\Users\\jairp\\Desktop\\BackUP\\CODE-20180719T021021Z-001\\CODE\\Datasets\\Cats and dogs\\PetImages"
CATEGORIES = ["Dog","Cat"]


# iterate through all of the examples of dog and cat 
for category in CATEGORIES: 
    path = os.path.join(DATADIR, category) # path to cats or dogs 
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # read image
        plt.imshow(img_array, cmap="gray") # show the image in grayscale
        plt.show()
        break
    break


print(img_array) # our image is our array 
print(img_array.shape)

# Notice that the photos are different shapes! 
# so we want to make everything the smae 
IMG_SIZE = 50 
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')

# Now let's create the training dataset

training_data = []

# note we got the features as numbers, but our classifications are 
# not numbers, so let dog = 1, cat = 0 
def create_training_data():
    for category in CATEGORIES: 
        path = os.path.join(DATADIR, category) # path to cats or dogs 
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass # skip the image
                

   
create_training_data()

print(len(training_data))

import random 

random.shuffle(training_data)

for sample in training_data: 
    print(sample[1]) # check that our labels are correct
   
X = [] # feature set
y = [] # labels

for features, label in training_data: 
    X.append(features)
    y.append(label)
    
# Need to convert to an np array
# -1 is how many features we have
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1 because it's greyscale

# grayscale convolutional NN

import pickle # to svae the data 

# save the data
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# load the data
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
print(X[1]) # feature

# **************************************************************************

### 3. Convolutional Neural Networks ### 


import time
import pickle as pk
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report


# algorithm

NAME = "Cats_vs_dogs_cnn_64x2{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


X = pk.load(open("X.pickle","rb")) 
y = pk.load(open("y.pickle","rb")) 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= .01, 
                                                    random_state=42)

# Scale according to the max # of pixels 

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()
# model.add(laye type, window, input_shape)

# Convolutional NN layer 
model.add( Conv2D(64, 3,3, input_shape = X.shape[1:]) ) # (50,50,1)
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2))) # window pool size 

# COnvolutional NN layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())  # flatten from 2D to 1D

#model.add(Dense(64)) # Dense layer
#model.add(Activation("relu"))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid')) # activation function 

# Model compilation 
model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics=['accuracy'])

# model fitting
model.fit(X_train, y_train, batch_size = 32, 
          epochs = 15, 
          validation_split = 0.3, 
          callbacks=[tensorboard]) # batch =  how many at the time


y_pred = np.sign(model.predict(X_test))
print(classification_report(y_test, y_pred))



# *****************************************************************************

### 5. Optimizin with TensorBoard ###


# Imports

import time
import pickle as pk
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report


# set up gpu 
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, 
                    layer_size, dense_layer, int(time.time()))
            
            print(NAME)
            
            tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))


            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])


# *****************************************************************************8
            
### Final model ### 
            
import time
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard 
from sklearn.model_selection import train_test_split 

# set up gpu 
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

conv_layer = 3
layer_size = 64
dense_layer = 0

NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, 
        layer_size, dense_layer + 1, int(time.time()))


X = pk.load(open("X.pickle","rb")) 
y = pk.load(open("y.pickle","rb")) 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= .25, 
                                                    random_state=42)

# Scale according to the max # of pixels 

X = X/255.0
X_train = X_train/255.0
X_test = X_test/255.0

tensorboard = TensorBoard(log_dir="final_logs\{}".format(NAME))

model = Sequential()

model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

for l in range(conv_layer-1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

for l in range(dense_layer):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          batch_size=32,
          epochs=7,
          validation_split=0.3,
          callbacks=[tensorboard])


loss, acc = model.evaluate(X_test, y_test)
print("Accuracy: {:5.2f}%  Loss: {:5.2f}".format(acc, loss))


# Save the model 
model.save('cats_dogs_3-conv-64-nodes-1-dense-1563666473.h5')

# Load the model 
new_model = keras.models.load_model('cats_dogs_3-conv-64-nodes-1-dense-1563666473.h5')
new_model.summary()

loss, acc = new_model.evaluate(X_test, y_test)
print("Accuracy: {:5.2f}%  Loss: {:5.2f}".format(acc, loss))


# ******************************************************************************8
