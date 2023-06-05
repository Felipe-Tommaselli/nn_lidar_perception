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

from sys import platform
import os
import cv2
import numpy as np
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
        if os.getcwd().split(SLASH)[-1] == 'src':
            os.chdir('../..') 

        # get the step number by the index
        step = self.labels.iloc[idx, 0]

        # get the path of the image
        path = os.getcwd() + SLASH + 'artificial_data' + SLASH + 'train' + SLASH
        full_path = os.path.join(path, 'image'+str(step)+'.png') # merge path and filename

        # TODO: IMPORT IMAGE WITH PIL
        # image treatment (only green channel)
        self.image = cv2.imread(full_path, -1)
        # to understand the crop, see the image in the assets folder and the lidar_tag.py file

        labels = self.labels.iloc[idx, 1:] # take step out of labels

        # labels = [m1, m2, b1, b2]
        m1, m2, b1, b2 = labels

        # PRE-PROCESSING
        # pre_process = PreProcess(dataset={'labels': labels, 'image': self.image})
        # labels, image = pre_process.pre_process()

        # labels_dep = PreProcess.deprocess(image=image, label=labels)        
        labels_dep = labels # just for now
        m1, m2, b1, b2 = labels_dep
        x1 = np.arange(0, image.shape[0], 1)
        x2 = np.arange(0, image.shape[0], 1)
        y1 = m1*x1 + b1
        y2 = m2*x2 + b2

        plt.plot(x1, y1, 'r')
        plt.plot(x2, y2, 'r')
        plt.imshow(image, cmap='gray')
        plt.show()

        return {"labels": labels, "image": image, "angle": 0}



############ DATA ############
batch_size = 96
csv_path = "../../artificial_data/tags/Artificial_Label_Data.csv"

dataset = ArtificialLidarDatasetCNN(csv_path)