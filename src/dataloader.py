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

from sys import platform
import os
import random
import cv2
import numpy as np
import math
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils.data import Dataset

global SLASH
if platform == "linux" or platform == "linux2":
    # linux
    SLASH = "/"
elif platform == "win32":
    # Windows...
    SLASH = "\\"

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


class LidarDatasetCNN(Dataset):
    ''' Dataset class for the lidar data with images. '''
    
    def __init__(self, csv_path, train=True):
        ''' Constructor of the class. '''
        
        self.labels = pd.read_csv(csv_path)
        self.train = train # bool

    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample image of the dataset. '''

        # move from root (\src) to \assets\images
        if os.getcwd().split(SLASH)[-1] == 'src':
            os.chdir('..') 

        # get the step number by the index
        step = self.labels.iloc[idx, 0]

        # get the path of the image
        path = os.getcwd() + SLASH + 'assets' + SLASH + 'train' + SLASH
        full_path = os.path.join(path, 'image'+str(step)+'.png') # merge path and filename

        # image treatment (only green channel)
        self.image = cv2.imread(full_path, -1)
        self.image = self.image[30:570, 30:570,1] # take only the green channel and crop the image
        # to understand the crop, see the image in the assets folder and the lidar_tag.py file

        labels = self.labels.iloc[idx, 1:] # take step out of labels

        # we need to invert the y axis, for some reason there is a difference between the 0,0 in matplotlib and cv2 
        plt.scatter(labels[0], 540 - labels[1], c='r', s=20)
        plt.scatter(labels[2], 540 - labels[3], c='r', s=20)
        plt.scatter(labels[4], 540 - labels[5], c='r', s=20)
        plt.scatter(labels[6], 540 - labels[7], c='r', s=20)
        plt.scatter(530, 530, c='g', s=30)

        azimuth1, azimuth2, intersec1, intersec2 = self.getLabels(idx=idx)
        labels = [azimuth1, azimuth2, intersec1, intersec2]


        # print the points of the labels
        # print('Labels: ', labels)
        # plot the image with matplotlib
        #plt.axis('off')
        #plt.imshow(self.image, cmap='gray')
        #plt.show()

        return {"labels": labels, "image": self.image}

    def getLabels(self, idx):
        ''' Returns the labels of the image. '''
        #         0 ,   1 ,  2  ,  3  ,  4  ,  5  ,  6  ,  7  
        # step, L_x0, L_y0, L_x1, L_y1, L_x2, L_y2, L_x3, L_y3
        labels = self.labels.iloc[idx, 1:] # take step out of labels

        # line equation: y = mx + b
        # m = (y2 - y1) / (x2 - x1)
        # b = y1 - m*x1
        
        # note that we need to prevent division by 0 exception
        # RuntimeWarning: divide by zero encountered in long_scalars
        if labels[2] - labels[0] == 0:
            labels[2] += 1
        if labels[6] - labels[4] == 0:
            labels[6] += 1

        m1 = (labels[3] - labels[1]) / (labels[2] - labels[0])
        m2 = (labels[7] - labels[5]) / (labels[6] - labels[4])
        
        b1 = labels[1] - m1*labels[0]
        b2 = labels[5] - m2*labels[4]

        print(f'm1 = labels[3] - labels[1] / labels[2] - labels[0] = {labels[3]} - {labels[1]} / {labels[2]} - {labels[0]} = {m1}')
        
        # azimuth1, azimuth2, intersec1, intersec2
        # angles in radians (azimuth1, azimuth2) and meters (intersec1, intersec2)
        return math.atan(m1), math.atan(m2), b1, b2



if __name__ == "__main__":
    # ld = LidarDataset(csv_path="~/Documents/IC_NN_Lidar/datasets/syncro_data_validation.csv")
    # print(ld.__getitem__(idx=0)["lidar"], len(ld.__getitem__(idx=0)["lidar"]))
    # print(ld.__getitem__(idx=0)["lidar"], len(ld.__getitem__(idx=10)["lidar"]))

    # print('-'*80)
    # ldCNN = LidarDatasetCNN(csv_path="~/Documents/IC_NN_Lidar/assets/tags/Label_Data.csv")
    # print(ldCNN.__getitem__(idx=0)['image'], len(ldCNN.__getitem__(idx=0)['image']))
    # print(ldCNN.__getitem__(idx=1)['image'], len(ldCNN.__getitem__(idx=1)['image']))
    pass