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
ORB SLAM run using DeepVO as motion model and ResNet-50 model for detecting loop closure.

### Dependencies
- [socket](https://docs.python.org/3/library/socket.html)
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)

### Usage: Python Server
```
python3 deepvo_server.py [SEQ_NUM]
python3 [LOOP CLOSURE SEVER GOES HERE]
```

### Usage: C++ Client
```
.\build.sh
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```


