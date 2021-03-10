'''
********************************************************************
* Program: hw2.py                                            
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
* Edge detection
* https://github.com/hualili/opencv/blob/master/deep-learning-2020S/5-Canny.py
* Draw Canny edge image back onto original image
* https://answers.opencv.org/question/55272/drawing-lines-from-canny/
* 
* Find contours
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
* https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
* Find largest contour
* https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
* Draw bounding box with text
* https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
*
* Notes:
* 1. Uses tf.keras included in Tensorflow 2.0 instead of separate Keras installation.
* 2. Need h5 file harryTest.h5 which has CNN trained on MNIST dataset.
* 3. Loads and deploys the CNN.             
********************************************************************
'''
from tensorflow.keras.models import load_model
import cv2

# previous video capture directly using frames from live video
'''
# video capture
import numpy as np

def video_capture():
    cap = cv2.VideoCapture(0)

    while(True):

        ret, frame = cap.read() 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,200)

        cv2.imshow('original',frame)
        cv2.imshow('gray scale',gray)
        cv2.imshow('Canny edges',edges)
        
        # Show Canny edges on original image
        # for bitwise_or the shape needs to be the same, so we need to add an axis, 
        # since our input image has 3 axis while the canny output image has only one 2 axis
        out = np.bitwise_or(frame, edges[:,:,np.newaxis])
        cv2.imshow('orig with edges', out)
        
        # Finding Contours 
        edges_copy = edges.copy()
        # Use a copy of the image e.g. edged.copy() 
        # since findContours alters the image 
        contours, hierarchy = cv2.findContours(edges_copy,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        # draw the biggest contour (c) in green
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
          
        cv2.imshow('Contours', frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()
'''
          
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

# Test code
# draw bounding box with predicted digit caption text

# find contours of a binary image
im = cv2.imread('sample_image.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# find the biggest countour (c) by the area
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
# draw the biggest contour (c) in green
image = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
# label rectangle with predicted digit caption text
cv2.putText(image, str(y_pred[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (36,255,12), 2)

# Resize image
imageResized = cv2.resize(image, (760, 540))                    
cv2.imshow('contours', imageResized)
# Press q to quit
cv2.waitKey(0) & 0xFF == ord('q')


# video capture saved as image frames, then image frames processed
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

# image_name: Image name
# returns: The image
def get_orig_image(image_name):
    img = cv2.imread(image_name, cv2.IMREAD_COLOR)
    return img
      
# Convert image frame to grayscale
# image_name: Image name
# returns: Grayscale image
def convert_image_to_grayscale(image_name):
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    return img

# gray_image: Grayscale image
# returns: Canny edge image, which is black and white
def get_canny_edges(gray_image):
    edges = cv2.Canny(gray_image,100,200)
    return edges
   
# img: The original image
# canny: Canny edges image 
# returns: Original image with Canny edges drawn over it
def get_orig_image_with_canny_edges(img, canny):
    # for bitwise_or the shape needs to be the same, so we need to add an axis, 
    # since our input image has 3 axis while the canny output image has only one 2 axis
    out = np.bitwise_or(img, canny[:,:,np.newaxis])
    return out

# Show the given image in a window with the given window name.
# window_name: Name of window image displayed in
# image: Image to show
# def show_image(window_name, image):
    # cv2.imshow(window_name, image)
    
# edges: Canny edges image
# returns: Contours found
def find_contours(edges):
    # Finding Contours 
    edges_copy = edges.copy()
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edges_copy,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours
    
# image: Original image frame
# contours: countours found from Canny edge image  
# return: Original image with all countours drawn on it
def find_all_contours(image, contours): 
    return cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 

# image: Original image frame
# contours: countours found from Canny edge image    
def find_biggest_contour(image, contours):
    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    # draw the biggest contour (c) in green
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    return image

# Get canny edge from one image frame
IMAGE_FRAME_NAME = "frame10.jpg"
img = get_orig_image(IMAGE_FRAME_NAME)
cv2.imshow("original", img)
gray_img = convert_image_to_grayscale(IMAGE_FRAME_NAME)
cv2.imshow("gray_image", gray_img)
canny_img = get_canny_edges(gray_img)
cv2.imshow("Canny edges", canny_img)
img_with_canny_edges = get_orig_image_with_canny_edges(img, canny_img)
cv2.imshow("orig with edges", img_with_canny_edges)
# Find and draw contours using Canny edges image
contours = find_contours(canny_img)
# img_with_all_contours = find_all_contours(img, contours)
# cv2.imshow("orig with all contours", img_with_all_contours)
img_biggest_contour = find_biggest_contour(img, contours)
cv2.imshow("orig with biggest contour", img_biggest_contour)
# Press any key to close windows
# waits indefinitely for a key stroke
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()