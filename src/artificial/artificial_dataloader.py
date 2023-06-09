#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class that loads the dataset for the neural network. 
- "class LidarDataset(Dataset)" loads the dataset from the csv file.
    This class with the raw lidar data it is used in the nn_mlp.py file (multi-layer perceptron Network).
- "class LidarDatasetCNN(Dataset)" loads the dataset from the images already processed from the lidar dataset.
    This class with the images (instead of the raw data) it is used in the nn_cnn.py file (convolutional Network).

@author: Felipe-Tommaselli
""" 
import warnings
warnings.filterwarnings("ignore")

from sys import platform
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import datasets
import torchvision.models as models

sys.path.append('../')
from pre_process import *

global SLASH
if platform == "linux" or platform == "linux2":
    # linux
    SLASH = "/"
elif platform == "win32":
    # Windows...
    SLASH = "\\"

class ArtificialLidarDatasetCNN(Dataset):
    ''' Dataset class for the lidar data with images. '''
    
    def __init__(self, csv_path):
        ''' Constructor of the class. '''
        self.labels = pd.read_csv(csv_path)

    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample image of the dataset. '''

        # move from root (\src) to \assets\images
        if os.getcwd().split(SLASH)[-1] == 'artificial':
            os.chdir('../..') 

        # get the step number by the index
        step = self.labels.iloc[idx, 0]
        # step = 300*(i - 1) + j, with i and j starting in 1
        i = int((step - 1) / 300) + 1
        j = (step - 1) % 300 + 1

        # get the path of the image
        path = os.getcwd() + SLASH + 'artificial_data' + SLASH + 'train' + SLASH
        full_path = os.path.join(path, 'image'+str(i)+ '_' + str(j) +'.png') # merge path and filename

        # TODO: IMPORT IMAGE WITH PIL
        # image treatment (only green channel)
        self.image = cv2.imread(full_path, -1)
        self.image = cv2.resize(self.image, (224, 224), interpolation=cv2.INTER_LINEAR)
        self.image = self.image[:, :, 1] # only green channel
        # to understand the crop, see the image in the assets folder and the lidar_tag.py file

        labels = self.labels.iloc[idx, 1:] # take step out of labels

        # labels = [m1, m2, b1, b2]
        m1, m2, b1, b2 = labels

        # correcting matplotlib and opencv axis problem
        m1 = -m1
        m2 = -m2
        b1 = 224 - b1
        b2 = 224 - b2
        labels = [m1, m2, b1, b2]

        # PRE-PROCESSING
        image = self.image # just for now
        
        labels = ArtificialLidarDatasetCNN.process_label(labels)
        
        # labels_dep = PreProcess.deprocess(image=self.image, label=labels)        
        # m1, m2, b1, b2 = labels_dep
        # x1 = np.arange(0, image.shape[0], 1)
        # x2 = np.arange(0, image.shape[0], 1)
        # y1 = m1*x1 + b1
        # y2 = m2*x2 + b2
        # plt.plot(x1, y1, 'r')
        # plt.plot(x2, y2, 'r')
        # plt.title(f'step={step}, i={i}, j={j}')
        # plt.imshow(image, cmap='gray')
        # plt.show()

        return {"labels": labels, "image": image, "angle": 0}

    @staticmethod
    def process_label(labels):
        ''' Process the labels to be used in the network. Normalize azimuth and distance intersection.'''
        DESIRED_SIZE = 224 #px
        MAX_M = 224 

        # NORMALIZE THE AZIMUTH 1 AND 2 
        m1 = labels[0]
        m2 = labels[1]

        # NORMALIZE THE DISTANCE 1 AND 2
        b1 = labels[2]
        b2 = labels[3]

        #! NORMALIZATION WITH m1, m2, b1, b2
        # # convert m1 to azimuth to angles
        # # y = m*x + b
        # # azimuth it is the angle of m1 in radians with atan 
        # # print('-'*20)
        # # print('m1:', m1)

        # #TODO: Find some way to normalize the azimuth without the atan problem             
        # # azimuth1 = np.arctan(m1)
        # # azimuth2 = np.arctan(m2)
        # # # print('azimuth1:', azimuth1)

        # # # normalize the azimuth (-pi to pi) -> (-1 to 1)
        # # azimuth1 = azimuth1 / np.pi
        # # azimuth2 = azimuth2 / np.pi        

        #* alternative without atan
        # azimuth1 = m1/DESIRED_SIZE  
        # azimuth2 = m2/DESIRED_SIZE

        # # since the data it is compatible to the image size we will relate as:
        # # image = IMAGE_WIDTH x IMAGE_HEIGHT
        # # as y = a*x + b -> b = y - a*x
        # # note that for this step we are already working with the cropped image
        # # so we will use DESRIED_SIZE
        # # and, the MAX_M for square cases is = DESIRED_SIZE
        # # -------------------------
        # # where the minimum distance is when:
        # # y = 0 and x = MAX_WIDTH with m = MAX_M
        # # so b = 0 - (MAX_M)*MAX_WIDTH <- minimum distance
        # MAX_M = DESIRED_SIZE
        # dmin = - MAX_M * DESIRED_SIZE
        # # and the maximum distance is when:
        # # y = MAX_HEIGTH and x = MAX_WIDTH with m = MIN_M
        # # so b = MAX_HEIGHT - (MIN_M)*MAX_WIDTH <- maximum distance
        # dmax = DESIRED_SIZE - (-MAX_M)*DESIRED_SIZE

        # # normalize the distance (-291600 to 292140) -> (-1 to 1)
        # b1 = 2*((b1 - dmin)/(dmax - dmin)) - 1
        # b2 = 2*((b2 - dmin)/(dmax - dmin)) - 1

        # return [azimuth1, azimuth2, b1, b2]

        #! NORMALIZATION WITH w1, w2, q1, q2
        
        # now we can normalize:
        b1 = b1 - 224 # matplotlib 0 its in the top
        b2 = b2 - 224
        
        w1, w2, q1, q2 = PreProcess.extract_label([m1, m2, b1, b2])

        q1 = q1/DESIRED_SIZE
        q2 = q2/DESIRED_SIZE
        w1 = w1/DESIRED_SIZE
        w2 = w2/DESIRED_SIZE

        return [w1, w2, q1, q2]



