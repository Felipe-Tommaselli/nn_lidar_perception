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

global MAX_WIDTH
global MAX_HEIGHT
global MAX_M
global MIN_M

MAX_WIDTH = 540
MAX_HEIGHT = 540
MAX_M = 540
MIN_M = -540

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
        # where the minimum distance is when:
        # y = 0 and x = MAX_WIDTH with m = MAX_M
        # so b = 0 - (MAX_M)*MAX_WIDTH <- minimum distance
        dmin = - MAX_M * MAX_WIDTH
        # and the maximum distance is when:
        # y = MAX_HEIGTH and x = MAX_WIDTH with m = MIN_M
        # so b = MAX_HEIGHT - (MIN_M)*MAX_WIDTH <- maximum distance
        dmax = MAX_HEIGHT - (MIN_M)*MAX_WIDTH

        # normalize the distance (-291600 to 292140) -> (-1 to 1)
        d1 = 2*((d1 - dmin)/(dmax - dmin)) - 1
        d2 = 2*((d2 - dmin)/(dmax - dmin)) - 1

        return [azimuth1, azimuth2, d1, d2]

    def process_image(self, images: np.array) -> np.array:
        ''' Returns the processed image. '''
        

        return images

    @staticmethod
    def deprocess(images, labels):
        ''' Returns the deprocessed image and label. '''

        # DEPROCESS THE LABEL

        # azimuths 1 e 2: tangent of the azimuth
        m1 = np.tan(np.pi * labels[0])
        m2 = np.tan(np.pi * labels[1])

        # distances 1 e 2: image borders normalization
        dmin = - MAX_M * MAX_WIDTH
        dmax = MAX_HEIGHT - (MIN_M)*MAX_WIDTH

        d1 = (dmax - dmin)*(labels[2] + 1)/2 + dmin
        d2 = (dmax - dmin)*(labels[3] + 1)/2 + dmin

        labels = [m1, m2, d1, d2]

        # DEPROCESS THE IMAGE
        images = images 

        return images, labels