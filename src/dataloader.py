#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class that loads the dataset for the neural network. 
- "class LidarDataset(Dataset)" loads the dataset from the csv file.
    This class with the raw lidar data it is used in the nn_mlp.py file (multi-layer perceptron Network).
- "class LidarDatasetCNN(Dataset)" loads the dataset from the images already processed from the lidar dataset.
    This class with the images (instead of the raw data) it is used in the nn_cnn.py file (convolutional Network).

@author: andres
@author: Felipe-Tommaselli
"""

import os
import random
import cv2
import numpy as np
import math
import pandas as pd
import torch

from torch.utils.data import Dataset

class LidarDataset(Dataset):
    ''' Dataset class for the lidar data. '''
    def __init__(self, csv_path, transform=None, target_transform=None) -> None:
        ''' Constructor of the class. '''
        
        self.labels = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample of the dataset. '''
        
        dl = self.labels.iloc[idx, 2]
        dr = self.labels.iloc[idx, 3]
        dratio = self.labels.iloc[idx, 4]
        heading = self.labels.iloc[idx, 1]
        lidar = np.empty(1081)
        for step in range(0,len(self.labels.iloc[idx,7:])):
            lidar[step] = (self.labels.iloc[idx, step+7])
        torch.from_numpy(lidar)
        self.sample = {"dl": dl , "dr": dr, "dratio":dratio, "heading": heading, "lidar": lidar}
        #print(self.sample)
        return self.sample

ld = LidarDataset(csv_path="~/Documents/IC_NN_Lidar/datasets/syncro_data_validation.csv")
print(ld.__getitem__(idx=0))
print(ld.__getitem__(idx=0)["lidar"], len(ld.__getitem__(idx=0)["lidar"]))
print(ld.__getitem__(idx=0)["lidar"], len(ld.__getitem__(idx=10)["lidar"]))


class LidarDatasetCNN(Dataset):
    ''' Dataset class for the lidar data with images. '''
    
    def __init__(self, img_path, csv_path, transform=None, target_transform=None):
        ''' Constructor of the class. '''
        
        self.image = None
        self.img_path = img_path
        self.labels = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample image of the dataset. '''

        step = self.labels.iloc[idx, 0]
        path = ''.join([self.img_path, str(step)+".png"])
        print('path:', path)
        self.image = cv2.imread(path, -1)
        azimuth1, azimuth2, intersec1, intersec2 = self.getLabels(idx=idx)
        sample = {"azimuth1": azimuth1, "azimuth2": azimuth2, "intersec1": intersec1, "intersec2": intersec2, "image": self.image}
        return sample

    def getLabels(self, idx):
        ''' Returns the labels of the image. '''
        #         0 ,   1 ,  2  ,  3  ,  4  ,  5  ,  6  ,  7  
        # step, L_x0, L_y0, L_x1, L_y1, L_x2, L_y2, L_x3, L_y3
        labels = self.labels.iloc[idx, 1:] # take step out of labels

        # line equation: y = mx + b
        # m = (y2 - y1) / (x2 - x1)
        # b = y1 - m*x1
        m1 = (labels[3] - labels[1]) / (labels[2] - labels[0])
        b1 = labels[1] - m1*labels[0]
        m2 = (labels[7] - labels[5]) / (labels[6] - labels[4])
        b2 = labels[5] - m2*labels[4]

        # azimuth1, azimuth2, intersec1, intersec2
        # angles in radians (azimuth1, azimuth2) and meters (intersec1, intersec2)
        return math.atan(m1), math.atan(m2), b1, b2

ldCNN = LidarDatasetCNN(img_path="~/Documents/IC_NN_Lidar/assets/images/image", csv_path="~/Documents/IC_NN_Lidar/assets/tags/Label_Data.csv")
print(len(ldCNN.__getitem__(idx=0)["image"]))
print(len(ldCNN.__getitem__(idx=10)["image"]))