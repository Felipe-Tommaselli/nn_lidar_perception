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

        self.labels = copy.deepcopy(dataset['labels'])
        self.image = copy.deepcopy(dataset['image'])
        
    def pre_process(self) -> list:
        ''' Returns the processed data. '''

        # PROCESS THE DATA
        self.labels = self.process_label(self.labels)
        self.image = self.process_image(self.image)

        return self.labels, self.image


    def process_label(self, labels: list) -> list:
        ''' Returns the processed label. '''

        # NORMALIZE THE AZIMUTH 1 AND 2 
        m1 = labels[0]
        m2 = labels[1]

        # convert m1 to azimuth to angles
        # y = m*x + b
        # azimuth it is the angle of m1 in radians with atan 
        azimuth1 = np.arctan(m1)
        azimuth2 = np.arctan(m2)

        # normalize the azimuth (-pi to pi) -> (-1 to 1)
        azimuth1 = azimuth1 / np.pi
        azimuth2 = azimuth2 / np.pi

        # NORMALIZE THE DISTANCE 1 AND 2
        d1 = labels[2]
        d2 = labels[3]

        # since the data it is compatible to the image size we will relate as:
        # image = IMAGE_WIDTH x IMAGE_HEIGHT
        # as y = a*x + b -> b = y - a*x
        # where the minimum distance is when y = 0 and x = 1 (max azimuth)
        return [azimuth1, azimuth2, d1, d2]

    def process_image(self, images: np.array) -> np.array:
        ''' Returns the processed image. '''
        

        return images