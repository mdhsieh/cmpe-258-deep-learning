#----------------------------------------------------------*
# program : mnist-cnn-train.py;          
# date    : Mar 4, 2021                                  
# ref: https://github.com/hualili/opencv/blob/master/deep-learning-2020S/20-2021S-0-7-1convnets-NumeralDet-saveTrained.py
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/            
#                                              
# purpose : demo of saving trained mnist net               
#----------------------------------------------------------

# Try larger CNN with dropout and more fully connected layers.
import tensorflow

#--------------build convnet--------------------*
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(30, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(15, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

#-----------flatten then 10-way classifier--------* 
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary() #check the model 

#-----------get NIST image data-------------------*
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#-------------show first 9 images---------* 
from matplotlib import pyplot

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_images[i], cmap=pyplot.get_cmap("gray"))
pyplot.show()


train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#-------------train----------------------*
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=200)

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc

#-------------save trained model---------* 
import h5py 
model.save('custom-mnist-cnn-v2.h5')
#-end 

'''
import tensorflow

#--------------build convnet--------------------*
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary() #check the model 

#-----------flatten then 10-way classifier--------* 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#-----------get NIST image data-------------------*
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#-------------show first 9 images---------* 
from matplotlib import pyplot

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_images[i], cmap=pyplot.get_cmap("gray"))
pyplot.show()


train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#-------------train----------------------*
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc

#-------------save trained model---------* 
import h5py 
model.save('harryTest.h5')
#-end 
'''

'''
# Try own images from custom dataset instead of mnist
import tensorflow

#--------------build convnet--------------------*
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(30, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(15, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

#-----------flatten then 10-way classifier--------* 
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary() #check the model 

import cv2
import glob
import numpy as np
import pandas as pd 

# Given an ROI image, create a new image with its largest dimension, ex.
# 211px height and 125px width image resized to 211x211 image.
# Purpose is to preserve aspect ratio.
# Then resize this image to 28x28. Convert to grayscale.
# image: ROI image
# returns: Square 28x28 image keeping ROI image's aspect ratio.
def get_resized_image(image):
    height, width, channels = image.shape
    # Create a background square image with 
    # size being the max dimension of ROI image
    maxDim = max(height, width)
    # black bg
    bg_img = np.zeros((maxDim, maxDim, 3), dtype = "uint8")
    bg_height, bg_width, channels = bg_img.shape
    
    # Use the ROI and background images' height and width
    # to place ROI image in center of background image.
    
    # compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round((bg_height-height)/2)
    xoff = round((bg_width-width)/2)

    # use numpy indexing to place the resized image in the center of background image
    result = bg_img.copy()
    result[yoff:yoff+height, xoff:xoff+width] = image
    
    # Resize the image to 28x28 pixels
    result_resized = cv2.resize(result, (28,28))
    # Convert to grayscale
    result_resized_gray = cv2.cvtColor(result_resized, cv2.COLOR_BGR2GRAY)
    
    return result_resized_gray

# Get images from matching string and place in list
# glob_string: Path with * image extension
# returns: List of images
def get_image_list(glob_string):
    filenames = [img for img in glob.glob(glob_string)]

    filenames.sort()

    images = []
    for img in filenames:
        # Read as grayscale to get 1 channel instead of 3
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        images.append(n)
        print(img)
    return images
    
# path: Path to CSV file
# column_name: Name of column which has labels
# returns list of labels
def get_labels_list(path, column_name):
    df = pd.read_csv(path, index_col=False)
    saved_column_as_list = df[column_name].tolist()
    return saved_column_as_list
   
from tensorflow.keras.utils import to_categorical

#-----------get NIST image data-------------------*
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
# Replace test mnist images with custom dataset test images
# 30 train images, 21 test images
# train_images = get_image_list("custom-dataset/train-images/*.png")
test_images = get_image_list("custom-dataset/test-images/*.png")
# train_labels = get_labels_list("custom-dataset/train-labels.csv", "train_label")
test_labels = get_labels_list("custom-dataset/test-labels.csv", "test_label")

from matplotlib import pyplot

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_images[i], cmap=pyplot.get_cmap("gray"))
    print(train_labels[i])
pyplot.show()

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(test_images[i], cmap=pyplot.get_cmap("gray"))
    print(test_labels[i])
pyplot.show()

# convert list to numpy array
# train_images = np.array(train_images)
test_images = np.array(test_images)

# print('train_images shape:', train_images.shape)
# print(type(train_images))
print('test_images shape:', test_images.shape)
print(type(train_images))

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((21, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=200)

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc

# save
import h5py 
model.save('custom-dataset-trained-mnist-cnn.h5')
'''