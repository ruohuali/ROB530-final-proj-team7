# Implementing Deep Learning based methods in SLAM algorithms
Team 7: Lounis Bouzit, Ruohua Li, Xun Tu, Ziyi Liu\
ROB 530 Final Project

ADD ABSTRACT HERE

## KITTI Dataset

## DeepVO
Helper files for training and running DeepVO model. We have outputted results in deepvo/poses for ease of use in other parts of our pipeline.
### Dependencies
- [PyTorch](https://pytorch.org/get-started/locally/)
- [torchvision](https://pytorch.org/get-started/locally/) 
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html) 
### Model & Path to Dataset
- Inside deepvo/params.py change self.data_dir to your path to KITTI dataset
- Add [pre-trained model](https://drive.google.com/file/d/1l0s3rYWgN8bL0Fyofee8IhN-0knxJF22/view) to deepvo/model (provided by [alexart13](https://github.com/alexart13))
### Usage
```
from deepvo_handler import DeepVOHandler
dvoh = DeepVOHandler('00') # KITTI sequence number 00
N = dvoh.get_len()
for i in range(N):
    rel_pose, abs_pose = dvoh.get_pose(i, prev_pose) # Gets pose relative to previous frame
```

## ResNet Loop Closure
Demo files to show how we have built, tested and deployed the ResNet-50 model.Including: demo of our network, demo of the usage of our model as a feature extractor, demo of the socket communication, demo of our own loop closing algorithm
### Dependencies
- [PyTorch](https://pytorch.org/get-started/locally/)
- [torchvision](https://pytorch.org/get-started/locally/) 
- [OpenCV] (https://opencv.org/)

### Usage
Please check the readme files contained in the related folders. 


## GTSAM Optimization
Two main functions called batch_main.py and incrm_main.py which use GTSAM optimizations on poses provided from DeepVO (use get_poses.py in deepvo/ to export).
### Dependencies
- [gtsam](https://pypi.org/project/gtsam/)
- [scipy](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html)

### Path to Dataset
- Change variable kitti_path to your path to KITTI dataset
- Uses deepvo/poses, ensure those are exported first with directions above

### Usage
```
python3 batch_main.py [SEQ_NUM]
python3 incrm_main.py [SEQ_NUM]
```

## ORB-SLAM2
ORB SLAM run using DeepVO as motion model for visual odometry.

### Dependencies
- [socket](https://docs.python.org/3/library/socket.html)
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)

### Usage: Python Server
```
python3 deepvo_server.py [SEQ_NUM]
```

### Usage: C++ Client
```
./build.sh
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```
## ORB-SLAM2_ResNet
ORB SLAM run using ResNet-50 as the feature extractor in loop detection part.
(We are separating the two pipelines because our models are kind of incompatible.
When using ResNet-50 as the feature extractor in loop detection task, the overall loop closing algorithm stays unchanged, so the whole loop closing task still depends on keyframes. However, our DeepVO model does not generate keyframes. Thus, we evaluate the performances separately. Also, this is part of the reason why we tried to develop our own SLAM pipeline trying to merge the two models together)

### Test Environment and Packages
System: Ubuntu 20.02 LTS
OpenCV: 3.2.0
Eigen: 3.2.10

### Dependencies
- [socket](https://docs.python.org/3/library/socket.html)
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)

### Usage: Python Server
Under ORB-SLAM2 directory, open the terminal and type
```
python3 server.py
```

### Usage: C++ Client
Open a separate window, and type
```
./build.sh
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```
as is explained in the readme file contained in ORB_SLAM2_resnet
### Others
1. In ORB-SLAM 2 folder, if the program is failed in running ./build.sh for ORB-SLAM 2: 
```
error: ‘decay_t’ is not a member of ‘std’ 
```
try changing "-std=c++11" to "-std=c++14" in CMakeList.txt

2. In ORB-SLAM 2 folder, if the program is failed saying that "no module 'Pangolin' or 'Eigen' is found", even if you have already installed them, try replacing the codes in CMakeList.txt
```
find_package(Eigen3 3.1.0 REQUIRED)
```
with 
```
list(APPEND CMAKE_INCLUDE_PATH "/usr/local/include")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
```
3. The files "resnet_server.py", "resnet50_places365.pth.tar", "categories_places365.txt" are just for you to have a brief preview on them. They are NOT working in ORB-SLAM2 pipeline. To see how to use them, please go into "ORB-SLAM2 ResNet" pipeline
