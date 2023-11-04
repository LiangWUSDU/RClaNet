# Source code for our paper entitled "RClaNet: An explainable Alzheimer’s disease diagnosis framework by joint deformable registration and classification"
joint registration and classification for early diagnosis of AD
The code was written by Dr. Liang Wu, Shandong University
We will publish tihs work, and when we publish it we will upload it here

# Requirements
System： Ubuntu 16.04 LTS with GeForce GTX TITAN X
Python: 3.6 Anaconda 64-bit
The framework needs the following dependencies:
tensorflow~=1.15.0
numpy~=1.21.5
keras~=2.1.4
scipy~=1.9.3
# How to exectue
Download the code to your own computer:

First, you need to run train_propose.py to train the joint registration and classification model.

Then run test_propose.py to get the global deformation field and disease risk probability map.

For the classification of AD, we consider three classification tasks. The classifiers are CNN and MHCNN. (train_CNN_CA.py, train_MHCNN_CA.py, et al.)

Finally, the trained model is used to test the test set. (CNN_CA_test.py,MHCNN_CA_test.py, et al.)
# Script that generates the H5 file is included
Please contact me if you need this file (wuliang@sdfmu.edu.cn).

There are others parameters can be set, please see the code for details.
# Ackonowleggment

Some codes in this repository are modified form Labreg and VoxelMorph.

# Contact:

If you have any problems or questions please contact(wuliang@sdfmu.edu.cn or hushunbo@lyu.edu.cn)
