### Changes
Improve detection speed made to original YOLOv4 repo by reducing number of classes.
#### Set up YOLOv4 and measured original detection speed
- Custom video being detected was data/video/road_traffic.mp4
- Program info is in detect_video.py
- In detect_video.py, ran video detection script: 
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/road_traffic.mp4 --output ./detections/results-original.avi
- Recorded original speed:
--- 278.4970302581787 seconds ---
#### Reduce number of classes
- New file data/classes/custom.names was created to list reduced number of classes.
Since this was traffic video listed only person and vehicles classes.
- Changed core/config.py to use the custom.names file.
- Then re-ran detection:
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/road_traffic.mp4 --output ./detections/results.avi
- Recorded new speed:
--- 277.21981978416443 seconds ---
#### Results
- Saved 1 second of total detection time
- The new detection results are in video/results.avi. Original results are in video/original-results.avi.

### Changed files
data/video/road_traffic.mp4
data/classes/custom.names
core/config.py
detect_video.py
detections/results-original.avi
detections/results.avi

### Instructions
##### Setup:
First create and activate Anaconda environment. Example using CPU.
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

Download yolov4.weights at:
https://drive.google.com/u/1/uc?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT&export=download

Then convert model weights from Darknet to TensorFlow.
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

##### Run:
conda activate yolov4-cpu
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video ./data/video/road_traffic.mp4 --output ./detections/results.avi

### Requirements
Anaconda 2020.11
Python 3.7.0
OpenCV-Python 4.1.1.26
TensorFlow 2.3.0

### More info:
Need to have all these packages in environment to run, view with:
conda list

# packages in environment at C:\Users\mdhsi\anaconda3\envs\yolov4-cpu:
#
# Name                    Version                   Build  Channel
absl-py                   0.12.0                   pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
blas                      1.0                         mkl
ca-certificates           2021.1.19            haa95532_1
cachetools                4.2.1                    pypi_0    pypi
certifi                   2020.12.5        py37haa95532_0
chardet                   4.0.0                    pypi_0    pypi
cycler                    0.10.0                   py37_0
easydict                  1.9                      pypi_0    pypi
freetype                  2.10.4               hd328e21_0
gast                      0.3.3                    pypi_0    pypi
google-auth               1.28.1                   pypi_0    pypi
google-auth-oauthlib      0.4.4                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
grpcio                    1.37.0                   pypi_0    pypi
h5py                      2.10.0                   pypi_0    pypi
hdf5                      1.8.20               hac2f561_1
icc_rt                    2019.0.0             h0cc432a_1
icu                       58.2                 ha925a31_3
idna                      2.10                     pypi_0    pypi
importlib-metadata        3.10.0                   pypi_0    pypi
intel-openmp              2020.2                      254
jpeg                      9b                   hb83a4c4_2
keras-preprocessing       1.1.2                    pypi_0    pypi
kiwisolver                1.3.1            py37hd77b12b_0
libopencv                 3.4.2                h20b85fd_0
libpng                    1.6.37               h2a8f88b_0
libtiff                   4.1.0                h56a325e_1
lxml                      4.6.3                    pypi_0    pypi
lz4-c                     1.9.3                h2bbff1b_0
markdown                  3.3.4                    pypi_0    pypi
matplotlib                3.3.4            py37haa95532_0
matplotlib-base           3.3.4            py37h49ac443_0
mkl                       2020.2                      256
mkl-service               2.3.0            py37h196d8e1_0
mkl_fft                   1.3.0            py37h46781fe_0
mkl_random                1.1.1            py37h47e9c7a_0
numpy                     1.18.5                   pypi_0    pypi
oauthlib                  3.1.0                    pypi_0    pypi
olefile                   0.46                     py37_0
opencv                    3.4.2            py37h40b0b35_0
opencv-python             4.1.1.26                 pypi_0    pypi
openssl                   1.1.1k               h2bbff1b_0
opt-einsum                3.3.0                    pypi_0    pypi
pillow                    8.2.0            py37h4fa10fc_0
pip                       21.0.1           py37haa95532_0
protobuf                  3.15.8                   pypi_0    pypi
py-opencv                 3.4.2            py37hc319ecb_0
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pyparsing                 2.4.7              pyhd3eb1b0_0
pyqt                      5.9.2            py37h6538335_2
python                    3.7.0                hea74fb7_0
python-dateutil           2.8.1              pyhd3eb1b0_0
qt                        5.9.7            vc14h73c81de_0
requests                  2.25.1                   pypi_0    pypi
requests-oauthlib         1.3.0                    pypi_0    pypi
rsa                       4.7.2                    pypi_0    pypi
scipy                     1.4.1                    pypi_0    pypi
setuptools                52.0.0           py37haa95532_0
sip                       4.19.8           py37h6538335_0
six                       1.15.0           py37haa95532_0
sqlite                    3.35.4               h2bbff1b_0
tensorboard               2.2.2                    pypi_0    pypi
tensorboard-plugin-wit    1.8.0                    pypi_0    pypi
tensorflow                2.3.0rc0                 pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
tf-estimator-nightly      2.3.0.dev2020062301          pypi_0    pypi
tk                        8.6.10               he774522_0
tornado                   6.1              py37h2bbff1b_0
tqdm                      4.60.0                   pypi_0    pypi
typing-extensions         3.7.4.3                  pypi_0    pypi
urllib3                   1.26.4                   pypi_0    pypi
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
werkzeug                  1.0.1                    pypi_0    pypi
wheel                     0.36.2             pyhd3eb1b0_0
wincertstore              0.2                      py37_0
wrapt                     1.12.1                   pypi_0    pypi
xz                        5.2.5                h62dcd97_0
zipp                      3.4.1                    pypi_0    pypi
zlib                      1.2.11               h62dcd97_4
zstd                      1.4.9                h19a0ad4_0