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
import json
import copy

sys.path.append('../')
from pre_process import *

global SLASH
if platform == "linux" or platform == "linux2":
    # linux
    SLASH = "/"
elif platform == "win32":
    # Windows...
    SLASH = "\\"

class NnDataLoader(Dataset):
    ''' Dataset class for the lidar data with images. '''
    
    def __init__(self, csv_path, train_path, runid):
        ''' Constructor of the class. '''
        labels = pd.read_csv(csv_path)
        self.train_path = train_path

        # move from root (\src) to \assets\images
        if os.getcwd().split(SLASH)[-1] == 'src':
            os.chdir('..') 

        self.images = list()
        self.labels_list = list()

        ############ LOAD DATASET ############

        for idx in range(len(labels)):

            # get the step number by the index
            step = labels.iloc[idx, 0]

            # full_path = os.path.join(path, 'image'+str(i)+ '_' + str(j) +'.png')
            full_path = os.path.join(self.train_path, 'image'+ str(step) +'.png') 

            # TODO: IMPORT IMAGE WITH PIL
            # image treatment (only green channel)
            image = cv2.imread(full_path, -1)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image[:, :, 1] # only green channel
            self.images.append(image)

            #* PROCESS LABELs
            wq_labels = NnDataLoader.process_label(labels.iloc[idx, 1:])

            self.labels_list.append(wq_labels) # take step out of labels

        ############ OBTAIN MEAN AND STD FOR NORMALIZATION ############
        
        # translate labels_list to numpy
        labels_numpy = np.asarray(self.labels_list)
        self.std = np.std(labels_numpy, axis=0)
        self.mean = np.mean(labels_numpy, axis=0)
        
        ############ SAVE MEAN AND STD IN config.yaml ############
        data = {
            'id': runid,  # You can set the appropriate id value
            'mean0': self.mean[0],
            'mean1': self.mean[1],
            'mean2': self.mean[2],
            'mean3': self.mean[3],
            'std0': self.std[0],
            'std1': self.std[1],
            'std2': self.std[2],
            'std3': self.std[3]
        }

        if os.getcwd() == 'src':
            os.chdir('..')
        filename = './models/params.json'

        # Load existing data if the file already exists
        try:
            with open(filename, 'r') as file:
                existing_data = json.load(file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            pass

        # Append the new data to the file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)


    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        
        return len(self.labels_list)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample image of the dataset. '''

        # PRE-PROCESSING
        image = copy.deepcopy(self.images[idx])
        #image = PreProcess.contours_image(image)  #! visual 

        labels = copy.deepcopy(self.labels_list[idx])
        labels = PreProcess.standard_extract_label(labels, self.mean, self.std)

        #? CHECKPOINT! Aqui as labels e a imagem estão corretas!!

        # labels_dep = PreProcess.standard_deprocess(image=image, label=labels, mean=self.mean, std=self.std)        
        # m1, m2, b1, b2 = labels_dep
        # x1 = np.arange(0, image.shape[0], 1)
        # x2 = np.arange(0, image.shape[0], 1)
        # y1 = m1*x1 + b1
        # y2 = m2*x2 + b2
        # plt.plot(x1, y1, 'r')
        # plt.plot(x2, y2, 'r')
        # plt.title(f'[Dataloader] step={idx}')
        # plt.imshow(image, cmap='gray')
        # plt.show()

        #! suppose m1 = m2
        w1, w2, q1, q2 = labels
        labels = [w1, q1, q2] # removing w2

        return {"labels": labels, "image": image, "angle": 0}

    @staticmethod
    def process_label(labels):
        ''' Process the labels to be used in the network. Normalize azimuth and distance intersection.'''
        DESIRED_SIZE = 224 #px
        MAX_M = 224 

        m1 = -labels[0]
        m2 = -labels[1]
        b1 = MAX_M - labels[2]
        b2 = MAX_M - labels[3]

        #! NORMALIZATION WITH w1, w2, q1, q2
        
        w1, w2, q1, q2 = PreProcess.parametrization(m1, m2, b1, b2)

        return [w1, w2, q1, q2]



