### Instructions
Run:
python hw2.py

Uses tf.keras included in Tensorflow 2.0 instead of separate Keras installation.

After running, several windows should display showing live video.
The frame called "orig with ROI box" should have bounding boxes with predicted digits.
Can show a paper with handwritten digits close to screen.

Demo videos: demo_video_1.webm, demo_video_2.webm, demo_video_3.webm

### Requirements
Python 3.6.6
CV2 4.5.1
Tensorflow 2.0.0

### Libraries
numpy

### More Info
The model is trained from mnist-cnn-train.py

hw2.py also accepts a video file as input.
Folders frames, rois, and rois-resized hold images saved
from a video file. Currently images are from sample_input_video.avi