#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class that preprocess the dataset for the neural network. The data it is loaded from the data_loader class.
At this point, the data intput should be the "data" from the LidarDatasetCNN class, before going into the Dataloader.
The data output should be the images and the labels (that crompise the dataset) after the pre-process is done. 

@author: Felipe-Tommaselli
"""

import copy

from sys import platform
import os
import cv2
import numpy as np
import math
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler

global SLASH
if platform == "linux" or platform == "linux2":
    # linux
    SLASH = "/"
elif platform == "win32":
    # Windows...
    SLASH = "\\"

class PreProcess:

    def __init__(self, dataset) -> None:
        ''' Constructor of the class. '''

        self.dataset = copy.deepcopy(dataset)

        print('-=-=-=--=-=-=-=-=-=-=-', self.dataset is dataset)
        self.labels = np.array([item['labels'] for item in dataset])
        self.labels = self.labels.reshape(-1, 4).tolist()
        # change labels shape back

        self.images = np.array([item['image'] for item in dataset])
        print(f'images shape: {self.images.shape}, len: {len(self.images)}')


    def pre_process(self) -> list:
        ''' Returns the processed data. '''

        # PROCESS THE DATA
        self.labels, stats = self.process_label(self.labels)
        self.images = self.process_image(self.images)

        # we want to return dataset in the same format as the original dataset
        # just with the processed data

        # self.dataset[0]['labels'] = self.labels[0]

        self.dataset[0] = 'zap'        
        print('self.labels[0]: ', self.labels[0])
        print('self.dataset[0]: ', self.dataset[0]['labels'])        
        # for i in range(0, len(self.dataset)):
        #     # TODO: check if this is the correct way to do it
        #     print('labels:', self.labels[i])
        #     print('before self.dataset[i]: ', self.dataset[i]['labels'])
            
        #     self.dataset[i]['labels'] = self.labels[i][:]

        #     print('after self.dataset[i]: ', self.dataset[i]['labels'])
        #     self.dataset[i]['image'] = self.images[i]

        print('***self.dataset[0]: ', self.dataset[0])
        return self.dataset, stats


    def process_label(self, labels: list) -> list:
        ''' Returns the processed label. '''

        # TODO: normalize the azimuth with atan

        label_norm = [[], [], [], []]
        for label in labels:
            label_norm[0].append(label[0])
            label_norm[1].append(label[1])
            label_norm[2].append(label[2])
            label_norm[3].append(label[3])

        # normalize the labels with sklearn minmax scaler
        scaler = MinMaxScaler()
        label_norm[0] = scaler.fit_transform(np.array(label_norm[0]).reshape(-1, 1))
        label_norm[1] = scaler.fit_transform(np.array(label_norm[1]).reshape(-1, 1))
        label_norm[2] = scaler.fit_transform(np.array(label_norm[2]).reshape(-1, 1))
        label_norm[3] = scaler.fit_transform(np.array(label_norm[3]).reshape(-1, 1))

        # get the mean and std of the labels
        mean = [np.mean(label_norm[0]), np.mean(label_norm[1]), np.mean(label_norm[2]), np.mean(label_norm[3])]
        std = [np.std(label_norm[0]), np.std(label_norm[1]), np.std(label_norm[2]), np.std(label_norm[3])]


        # change to the old format: [len, len, len, len] -> len*[1, 1, ,1 ,1]
        label = []
        for i in range(len(label_norm[0])):
            # [0] -> azimuth 1, [1] -> azimuth 2, [2] -> intersec 1, [3] -> intersec 2
            # [i] -> the ith element of the list 
            # [0] -> the only element in the nparray (the value)
            label.append([label_norm[0][i][0], label_norm[1][i][0], label_norm[2][i][0], label_norm[3][i][0]])

        # print('-'*65)
        # print('labels after normalization: ', label_norm[0][:3])
        # print(f'labels len: {len(label_norm)}, labels 0 shape: {label_norm[0].shape}')
        # # print(f'mean: {mean}, std: {std}')
        # print('labels: ', label[:5])
        # print('-'*65)

        label = [[0, 0, 0, 0] for i in range(len(label))]
        print('label: ', label[:5])
        return label, {'mean': mean, 'std': std}

    def process_image(self, images: np.array) -> np.array:
        ''' Returns the processed image. '''
        

        return images