#----------------------------------------------------------*
# program : mnist-cnn-train.py;          
# date    : Mar 4, 2021                                  
# ref: https://github.com/hualili/opencv/blob/master/deep-learning-2020S/20-2021S-0-7-1convnets-NumeralDet-saveTrained.py
#                                                          
# purpose : demo of saving trained mnist net               
#----------------------------------------------------------

# Try own images from custom dataset instead of mnist
import tensorflow

from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

from tensorflow.keras.utils import to_categorical

import cv2
import glob
import numpy as np
import pandas as pd 

# Get images from matching string and place in list
# glob_string: Path with * image extension
# returns: List of images
def get_image_list(glob_string):
    filenames = [img for img in glob.glob(glob_string)]

    filenames.sort()

    images = []
    for img in filenames:
        # n = cv2.imread(img)
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
    
train_images = get_image_list("custom-dataset/train-images/*.png")
test_images = get_image_list("custom-dataset/test-images/*.png")
train_labels = get_labels_list("custom-dataset/train-labels.csv", "train_label")
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
train_images = np.array(train_images)
test_images = np.array(test_images)

print('train_images shape:', train_images.shape)
print(type(train_images))
print('test_images shape:', test_images.shape)
print(type(train_images))

# 30 train images, 9 test images
train_images = train_images.reshape((30, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((9, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=3)

test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc

# save
import h5py 
model.save('custom-mnist-cnn.h5')

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

