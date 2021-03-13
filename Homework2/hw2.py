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
* Save image frames
* https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
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
* Resize image
* https://stackoverflow.com/questions/19098104/python-opencv2-cv2-wrapper-to-get-image-size
* https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
* Create black image
* https://stackoverflow.com/questions/40901906/create-a-black-image-with-specified-size-depth-channels-with-opencv-cv2-module
* Place image in center of another
* https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
*
* Notes:
* 1. Uses tf.keras included in Tensorflow 2.0 instead of separate Keras installation.
* 2. Need h5 file harryTest.h5 which has CNN trained on MNIST dataset.
* 3. Loads and deploys the CNN.             
********************************************************************
'''
from tensorflow.keras.models import load_model
import cv2

# do bitwise or to show Canny edges on original image
import numpy as np
          
# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

'''
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
'''
    
# Reshape image array into format that model needs to make prediction
# img: Image array
def reshape_image(img):
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

# video capture saved as image frames, then image frames processed

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
    
# edges: Canny edges image
# returns: Contours found
def find_contours(edges):
    # Finding Contours 
    edges_copy = edges.copy()
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image.
    # Retrieve external contours only
    contours, hierarchy = cv2.findContours(edges_copy,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours
    
# Given an ROI image, create a new image with its largest dimension, ex.
# 211px height and 125px width image resized to 211x211 image.
# Purpose is to preserve aspect ratio.
# Then resize this image to 28x28. Convert to grayscale.
# image: ROI image
# returns: Square 28x28 image keeping ROI image's aspect ratio.
def get_resized_image(image):
    height, width, channels = image.shape
    # Create a black background square image with 
    # size being the max dimension of ROI image
    maxDim = max(height, width)
    black_img = np.zeros((maxDim, maxDim, 3), dtype = "uint8")
    bg_height, bg_width, channels = black_img.shape
    
    # Use the ROI and black images' height and width
    # to place ROI image in center of black image.
    
    # compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round((bg_height-height)/2)
    xoff = round((bg_width-width)/2)

    # use numpy indexing to place the resized image in the center of background image
    result = black_img.copy()
    result[yoff:yoff+height, xoff:xoff+width] = image
    
    # Resize the image to 28x28 pixels
    result_resized = cv2.resize(result, (28,28))
    # Convert to grayscale
    result_resized_gray = cv2.cvtColor(result_resized, cv2.COLOR_BGR2GRAY)
    
    return result_resized_gray
    
    
# Get bounding box from contour, then get 
# Region Of Interest (ROI) image from bounding box. 
# Resize the ROI image and predict digit.
# Then draw bounding box with predicted digit labeled on it.
# image: Original image frame
# contours: countours found from Canny edge image 
# returns: Copy of original image with bounding box
def get_bounding_box_image(image, contours):
    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    
    ROI_number = 0
    copy = image.copy()
    # for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    
    ROI = image[y:y+h, x:x+w]
    # Save ROI image
    # cv2.imwrite('rois/ROI_{}.png'.format(ROI_number), ROI)
    
    # resize image and save
    resized_img = get_resized_image(ROI)
    # cv2.imwrite('rois-resized/resized_ROI_{}.png'.format(ROI_number), resized_img)
    
    # Use the resized ROI image to predict the digit inside ROI.
    
    # load the image
    # imgToArr = load_image('rois-resized/resized_ROI_{}.png'.format(ROI_number)) 
    
    imgToArr = reshape_image(resized_img)
    
    # predict the digit
    y_pred = model.predict_classes(imgToArr)
    digit = y_pred[0]
    # print(digit)
    
    ROI_number += 1
    
    copy = cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),2)
    # label rectangle with predicted digit caption text
    cv2.putText(copy, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (36,255,12), 2)
    return copy

# Main
'''
# Get bounding boxes with predicted digit from one image frame
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
# Get ROI image, predict digit, and draw original bounding box with labeled prediction.
img_with_roi_bounding_box = get_bounding_box_image(img, contours)
cv2.imshow("orig with ROI box", img_with_roi_bounding_box)
# Press any key to close windows
# waits indefinitely for a key stroke
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
'''

# Get bounding boxes with predicted digit in live video capture
def video_capture():
    cap = cv2.VideoCapture(0)

    while(True):

        ret, img = cap.read() 

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny_img = get_canny_edges(gray_img)
        img_with_canny_edges = get_orig_image_with_canny_edges(img, canny_img)

        cv2.imshow('orig',img)
        cv2.imshow('gray scale',gray_img)
        cv2.imshow('Canny edges',canny_img)
        cv2.imshow('orig with edges', img_with_canny_edges)
        
        contours = find_contours(canny_img)
        
        img_with_roi_bounding_box = get_bounding_box_image(img, contours)
        
        cv2.imshow("orig with ROI box", img_with_roi_bounding_box)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()
    
video_capture()