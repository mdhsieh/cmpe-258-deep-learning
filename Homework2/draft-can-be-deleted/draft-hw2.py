'''
********************************************************************
* Program: draft-hw2.py                                            
* Coded by: Michael Hsieh                                                      
* Date: Mar 4 2021
*                                                                  
* References: 
* Predict image
* https://github.com/hualili/opencv/blob/master/deep-learning-2020S/20-2021S-3-load-deployment.py 
* https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
* Video capture
* https://github.com/hualili/opencv/blob/master/deep-learning-2020S/4-pvideoFile2019-1-30.py
* https://github.com/hualili/opencv/blob/master/deep-learning-2020S/3-pvideoCAM2019-1-30.py
* https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
* Save image frames
* https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
* Edge detection
* https://github.com/hualili/opencv/blob/master/deep-learning-2020S/5-Canny.py
* Draw Canny edge image back onto original image
* https://answers.opencv.org/question/55272/drawing-lines-from-canny/
* 
* Notes:
* 1. Uses tf.keras included in Tensorflow 2.0 instead of separate Keras installation.
* 2. Need h5 file harryTest.h5 which has CNN trained on MNIST dataset.
* 3. Loads and deploys the CNN.             
********************************************************************
'''
from tensorflow.keras.models import load_model
import cv2

# video capture
import numpy as np

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)

    cv2.imshow('original',frame)
    cv2.imshow('gray scale',gray)
    cv2.imshow('Canny edges',edges)
    
    # for bitwise_or the shape needs to be the same, so we need to add an axis, 
    # since our input image has 3 axis while the canny output image has only one 2 axis
    out = np.bitwise_or(frame, edges[:,:,np.newaxis])
    cv2.imshow('orig with edges', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
      
# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load model
model = load_model('harryTest.h5')
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
                
# load the image
imgToArr = load_image('sample_image.png')  

# predict the class
y_pred = model.predict_classes(imgToArr)
print(y_pred)              
               
# Tester code
'''
# video capture
import numpy as np

# output_video_name: Name of the video saved from capture
def capture_video(output_video_name):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):

        ret, frame = cap.read() 

        # write the image frame
        out.write(frame)

        cv2.imshow('original',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Get image frames from captured video
# input_video_name: Name of the video to save image frames from
def save_image_frames(input_video_name):
    videoName = "output-3.avi"
    cap = cv2.VideoCapture(videoName)
    ret, frame = cap.read()
        
    count = 0
    while success:
      # save 1 frame per second
      cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
      # save frame as JPEG file      
      cv2.imwrite("frame%d.jpg" % count, frame)
      ret, frame = cap.read()
      print('Read a new frame: ', ret)
      count += 1    

def get_orig_image(image_name):
    window_name = "orig"
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    cv2.imshow(window_name, img)
    # waits indefinitely for a key stroke
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return img
      
# Convert image frame to grayscale
# image_name: Image name
def convert_image_to_grayscale(image_name):
    window_name = "gray_image"
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow(window_name, img)
    # waits indefinitely for a key stroke
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return img

gray_img = convert_image_to_grayscale("frame14.jpg")

def get_canny_edges(gray_image):
    window_name = "Canny edges"
    edges = cv2.Canny(gray_image,100,200)

    cv2.imshow(window_name,edges)
    # waits indefinitely for a key stroke
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return edges
  
canny_img = get_canny_edges(gray_img)

def get_orig_image_with_canny_edges(img, canny):
    window_name = "orig with edges"
    # for bitwise_or the shape needs to be the same, so we need to add an axis, 
    # since our input image has 3 axis while the canny output image has only one 2 axis
    out = np.bitwise_or(img, canny[:,:,np.newaxis])
    cv2.imshow(window_name, out)
    # waits indefinitely for a key stroke
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return out

img = get_orig_image("frame14.jpg")
get_orig_image_with_canny_edges(img, canny_img)
'''

