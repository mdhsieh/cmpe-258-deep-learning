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
* Get all images in directory
* https://stackoverflow.com/questions/38675389/python-opencv-how-to-load-all-images-from-folder-in-alphabetical-order
*
* Larger CNN
* https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
*
* opencv morphological operations
* https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
* mnist sample image
* https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
*
* Notes:
* 1. Uses tf.keras included in Tensorflow 2.0 instead of separate Keras installation.
* 2. Need h5 file custom-mnist-cnn-v2.h5 which has CNN trained on MNIST dataset.
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
    
# Reshape image array into format that model needs to make prediction
# img: Image array
def reshape_image(img):
    # OpenCV imread reads image in BGR format but keras reads in RGB as PIL image
    # reverse channels to get RGB
    img = img[...,::-1]
    
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load model
# Try larger CNN with dropout and more fully connected layers.
model = load_model('custom-mnist-cnn-v2.h5')
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

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
   
# Create black background image.
# bg_height: Background image height
# bg_width: Background image width
# Returns: A background image which has 1 channel, meaning black and white image
def create_background_image(bg_height, bg_width):
    # black bg
    return np.zeros((bg_height, bg_width, 1), dtype = "uint8")
    
# Place an image in center of a background image
# preconditon: The images have 1 channel, meaning black and white.
# The background image is larger than the image to be centered.
# img: The image to place in center
# bg_img: Background image
def create_image_centered_in_background(img, bg_img):
    height, width, channels = img.shape
    bg_height, bg_width, channels = bg_img.shape
    # compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round((bg_height-height)/2)
    xoff = round((bg_width-width)/2)

    # use numpy indexing to place the resized image in the center of background image
    result = bg_img.copy()
    result[yoff:yoff+height, xoff:xoff+width] = img
    return result
   
# Given an ROI image, resize the image to 20x20x1
# and place it in a 28x28x1 background image.
# The final ROI image will be resized to 28x28 pixels.
# image: ROI image
# returns: Square 28x28 image.
def get_resized_image(image):
    # perform dilation to make white digit larger so it is easier to read
    kernel = np.ones((5,5),np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    
    # Keep aspect ratio
    # by creating background image with ROI image's max dimension,
    # then placing ROI in center of that image
    height, width = image.shape
    # make 1 channel
    image = image.reshape(height, width, 1)
    maxDim = max(height, width)
    # Create background square image
    bg_image = create_background_image(maxDim, maxDim)
    # place ROI image in center of background image
    keep_aspect_img = create_image_centered_in_background(image, bg_image)
    
    # resize image to 20x20x1
    image = cv2.resize(keep_aspect_img, (20,20), interpolation = cv2.INTER_AREA)
    image = image.reshape(20, 20, 1)
    
    # Create a background square image with 
    # size 28x28, so there are 8 bits of padding between background and digit
    bg_image = create_background_image(28, 28)
    
    # Use the ROI and background images' height and width
    # to place ROI image in center of background image.
    result = create_image_centered_in_background(image, bg_image)

    return result
    
# Get bounding box from contour, then get 
# Region Of Interest (ROI) image from bounding box. 
# Resize the ROI image and predict digit.
# Then draw bounding box with predicted digit labeled on it.
# image: Canny edge image to get ROI image from
# contours: countours found from Canny edge image 
# orig_image: Original image frame
# frame_num: Image frame number, used to create filenames of frame's ROIs.
# Default no frame number.
# save_rois: Whether to save ROI and resized ROI images in folders or not
# Default don't save.
# returns: Copy of original image with bounding box
def get_bounding_box_image(image, contours, orig_image, frame_num = None, save_rois = False):
    
    ROI_FOLDER = "rois/"
    RESIZED_ROI_FOLDER = "rois-resized/"
    
    # Set a minimum area so unecessary contours are eliminated
    MIN_AREA = 50
    
    ROI_number = 0
    copy = orig_image.copy()
    for c in contours:
        if cv2.contourArea(c) > MIN_AREA:
            x,y,w,h = cv2.boundingRect(c)
            
            ROI = image[y:y+h, x:x+w]
            if save_rois:
                # Save ROI image
                roi_path = ROI_FOLDER + frame_num + "_" + "ROI_{}.png".format(ROI_number)
                cv2.imwrite(roi_path, ROI)
                print("saved " + roi_path)
            
            # resize image
            resized_img = get_resized_image(ROI)
            if save_rois:
                # Save resized ROI image
                resized_roi_path = RESIZED_ROI_FOLDER + frame_num + "_" + "resized_ROI_{}.png".format(ROI_number)
                cv2.imwrite(resized_roi_path, resized_img)
                print("saved " + resized_roi_path)
            
            # Use the resized ROI image to predict the digit inside ROI.
            
            # load the image
            imgToArr = reshape_image(resized_img) 
                        
            # predict the digit
            y_pred = model.predict_classes(imgToArr)
            digit = y_pred[0]
            
            # Increase saved ROI number by 1
            ROI_number += 1
            
            copy = cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),2)
            # label rectangle with predicted digit caption text
            cv2.putText(copy, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (36,255,12), 2)
    return copy


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
        
        img_with_roi_bounding_box = get_bounding_box_image(canny_img, contours, img)
        
        cv2.imshow("orig with ROI box", img_with_roi_bounding_box)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()

# Main
video_capture()


# Comment out main  video_capture() function and uncomment this block to 
# save an input video as image frames,
# save the ROIs and resized ROIs from each image frame, and then
# display each frame with labeled bounding boxes.
# Current folders have frames and ROIs from sample_input_video.avi
'''
# Capture live video, then save the video with given name.
# Using .avi extension
# output_video_name: Name of the video saved from capture
def capture_video(saved_video_name):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(saved_video_name,fourcc, 20.0, (640,480))

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
    videoName = input_video_name
    cap = cv2.VideoCapture(videoName)
    success, frame = cap.read()
        
    count = 0
    while success:
      # save 1 frame per second
      cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
      # num is 4 characters long with leading 0
      filename = "frames/frame%s.jpg" % str(count).zfill(4)
      # save frame as JPEG file      
      cv2.imwrite(filename, frame)
      success, frame = cap.read()
      print('Read a new frame: ', success)
      count += 1    

# get image frame number
import os

# sort filenames
import glob

# Uncomment to capture a video, save it, then save new image frames from it.
# This will overwrite any existing frames and ROI images.
# video downloaded from capture saved as image frames, then image frames processed
# VIDEO_NAME = "sample_input_video.avi"
# capture_video(VIDEO_NAME) 
# save_image_frames(VIDEO_NAME)   

filenames = [img for img in glob.glob("frames/frame*.jpg")]

# sort image frames in order
filenames.sort()

images = []
for img in filenames:
    n= cv2.imread(img)
    images.append(n)
    print(img)
    
for img in filenames:
    # Get bounding boxes with predicted digit from one image frame
    # for example "frames/frame0000.jpg"
    IMAGE_FRAME_NAME = img
    
    frame_name_without_extensions = os.path.splitext(IMAGE_FRAME_NAME)[0]
    FRAME_NUM = frame_name_without_extensions.split("frames",1)[1] 
    
    img = cv2.imread(IMAGE_FRAME_NAME, cv2.IMREAD_COLOR)
    cv2.imshow("original", img)
    gray_img = cv2.imread(IMAGE_FRAME_NAME, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("gray image", gray_img)
            
    canny_img = get_canny_edges(gray_img)

    cv2.imshow("Canny edges", canny_img)
    img_with_canny_edges = get_orig_image_with_canny_edges(img, canny_img)
    cv2.imshow("orig with edges", img_with_canny_edges)
    # Find and draw contours using Canny edges image
    contours = find_contours(canny_img)
    # Get ROI image, predict digit, and draw original bounding box with labeled prediction.
    img_with_roi_bounding_box = get_bounding_box_image(canny_img, contours, img, FRAME_NUM, True)
    cv2.imshow("orig with ROI box", img_with_roi_bounding_box)
    # Press any key to close windows
    # waits indefinitely for a key stroke
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
'''