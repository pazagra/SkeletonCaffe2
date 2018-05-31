# SkeletonCaffe2

This respository aims to provide a caffe2 version in python of the work:

- *Realtime Multi-Person Pose Estimation*. By Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## Contents
1. [Installation](#installation)
2. [Usage](#usage)

## Installation

This version needs caffe2, Opencv and scipy to work. Then clone this repository:

```
git clone https://github.com/pazagra/SkeletonCaffe2'
```

After that you need to download and convert the pretrained models:
```
cd model
chmod +x get_model.sh
./get_model.sh
```

This will populate the model folder.

## Usage
You can launch the Test provided:

```
python TestC2.py
```
This should print the dictionary of the skeleton and show the image with the skeleton drawn.

The code for the skeleton is in the file CNNCafe2.py and it is encapsulate in the class skeleton. This class has a function call get_skeleton that receives a images and the scales that you want to use (the lower the scale is, the faster it will be but it will have less accuracy).

#Limitations

For the moment, it is limited to output one skeleton for each image. The Bounding Box is calculated when more than one skeleton is found and the bigger one is return.
