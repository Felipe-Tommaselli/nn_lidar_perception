#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The class that loads the dataset for the neural network. 
This data loader loads all data from the "Images" folder and the specific labels from the ".csv" file.
With that, two processes take place: Data normalization and Dataset creation. 
1. Data normalization: The normalization with the best results is the "standard": (xi - mean(x)) / (std(x))
    - the mean an std used for each label is stored in the params.json per "ruind" as an identifier
    - this distribution guarantees mean = 0 and std = 1, with a balanced dataset 
2. Dataset creation succeed through the __get_item__() called from the main.py
    - it is worth noticing that there are two options: load each image at each __get_item__() call or 
    load the entire image dataset. The second option consumes a lot of RAM memory but is faster during the training. 

Difference from 'dataloader.py' and 'test_dataloader.py'
dataloader.py: loads the entire image dataset once
test_dataloader.py: load each image at each get_item call

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

from pre_process import *

# move from root (\src) to \assets\images
if os.getcwd().split(r'/')[-1] == 'src':
    os.chdir('..') 

class NnDataLoader(Dataset):
    ''' Dataset class for the lidar data with images. '''
    
    def __init__(self, csv_path, train_path, runid):
        ''' Constructor of the class. '''
        labels = pd.read_csv(csv_path)
        self.train_path = train_path
        self.images = list()
        self.labels_list = list()

        ############ LOAD DATASET ############
        for idx in range(len(labels)):
            step = labels.iloc[idx, 0] # step number by the index
            full_path = os.path.join(self.train_path, 'image'+ str(step) +'.png') 

            ############ PROCESS IMAGE ############
            image = cv2.imread(full_path, -1)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image[:, :, 1] # only green channel
            self.images.append(image)

            ############ PROCESS LABEL ############
            wq_labels = NnDataLoader.process_label(labels.iloc[idx, 1:])
            self.labels_list.append(wq_labels) # take step out of labels

        ############ OBTAIN MEAN AND STD FOR NORMALIZATION ############
        labels_numpy = np.asarray(self.labels_list)
        self.std = np.std(labels_numpy, axis=0)
        self.mean = np.mean(labels_numpy, axis=0)
        
        ############ SAVE MEAN AND STD IN "params.json" ############
        new_params = {
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

        with open(filename, 'r') as file:
            existing_data = json.load(file)

        existing_data.append(new_params)
        with open(filename, 'w') as file:
            json.dump(existing_data, file, indent=4)


    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        return len(self.labels_list)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample image of the dataset. '''
        image = copy.deepcopy(self.images[idx])
        labels = copy.deepcopy(self.labels_list[idx])
        
        labels = PreProcess.standard_extract_label(labels, self.mean, self.std)
        #image = PreProcess.contours_image(image)  #! visual, bad results 

        #? CHECKPOINT!

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
        IMG_SIZE = 224 #px

        m1 = -labels[0]
        m2 = -labels[1]
        b1 = IMG_SIZE - labels[2]
        b2 = IMG_SIZE - labels[3]

        # obs: IMG_SIZE change matplotlib and opencv start y to the origin 

        # NORMALIZATION WITH w1, w2, q1, q2        
        w1, w2, q1, q2 = PreProcess.parametrization(m1, m2, b1, b2)
        return [w1, w2, q1, q2]




class TestNnDataLoader(Dataset):
    ''' Dataset class for the lidar data with images. '''
    
    def __init__(self, csv_path, train_path):
        ''' Constructor of the class. '''
        self.labels = pd.read_csv(csv_path)
        self.train_path = train_path

    def __len__(self) -> int:
        ''' Returns the length of the dataset (based on the labels). '''
        
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        ''' Returns the sample image of the dataset. '''

        # move from root (\src) to \assets\images
        if os.getcwd().split(r'/')[-1] == 'src':
            os.chdir('..') 

        # get the step number by the index
        step = self.labels.iloc[idx, 0]

        # full_path = os.path.join(path, 'image'+str(i)+ '_' + str(j) +'.png') # merge path and filename
        full_path = os.path.join(self.train_path, 'image'+ str(step) +'.png') # merge path and filename


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

        #? CHECKPOINT! Aqui as labels e a imagem estão corretas!!

        # PRE-PROCESSING
        image = self.image # just for now
        #image = PreProcess.contours_image(image)  #! visual 

        #* PROCESS LABELs
        labels = TestNnDataLoader.process_label(labels)

        # labels_dep = PreProcess.deprocess(image=self.image, label=labels)        
        # m1, m2, b1, b2 = labels_dep
        # x1 = np.arange(0, image.shape[0], 1)
        # x2 = np.arange(0, image.shape[0], 1)
        # y1 = m1*x1 + b1
        # y2 = m2*x2 + b2
        # plt.plot(x1, y1, 'r')
        # plt.plot(x2, y2, 'r')
        # plt.title(f'[Dataloader] step={step}')
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

        m1 = labels[0]
        m2 = labels[1]
        b1 = labels[2]
        b2 = labels[3]

        #! NORMALIZATION WITH w1, w2, q1, q2
        
        w1, w2, q1, q2 = PreProcess.extract_label([m1, m2, b1, b2])

        return [w1, w2, q1, q2]


