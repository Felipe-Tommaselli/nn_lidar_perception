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
global CROP_FACTOR
global RESIZE_FACTOR

MAX_WIDTH = 540
MAX_HEIGHT = 540
MAX_M = 540
MIN_M = -540 
CROP_FACTOR_Y = 0.7 #%
CROP_FACTOR_X = 0.05 #%
RESIZE_FACTOR = 0.5

class PreProcess:

    def __init__(self, dataset) -> None:
        ''' Constructor of the class. '''

        self.labels = copy.deepcopy(dataset['labels'])
        self.image = copy.deepcopy(dataset['image'])
        
    def pre_process(self) -> list:
        ''' Returns the processed data. '''

        self.image = self.process_image(self.image, self.labels)
        self.labels = self.process_label(self.labels)

        return self.labels, self.image

    def process_image(self, image: np.array, labels) -> np.array:

        ''' Returns the processed image. '''
        MAX_WIDTH = image.shape[0]
        MAX_HEIGHT = image.shape[1]

        # Crop the image to the region of interest
        cropped_size_y = int(MAX_HEIGHT * CROP_FACTOR_Y)
        cropped_size_x = int(MAX_WIDTH * CROP_FACTOR_X)
        cropped_image = image[-cropped_size_y:, cropped_size_x:-cropped_size_x]

        # Resize the image to a smaller size
        resized_image = cv2.resize(cropped_image, (int(MAX_WIDTH*RESIZE_FACTOR), int(MAX_HEIGHT*RESIZE_FACTOR)))

        return resized_image

    def process_label(self, labels: list) -> list:
        ''' Returns the processed label. '''

        # scale the labels acording to the image size
        labels = PreProcess.scale_labels(labels)

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

    # UTILITIES FUNCTIONS FOR DEPROCESSING AND ROUTINE OPERATIONS

    @staticmethod
    def deprocess(image, label):
        ''' Returns the deprocessed image and label. '''

        # DEPROCESS THE LABEL

        # azimuths 1 e 2: tangent of the azimuth
        m1 = np.tan(np.pi * label[0])
        m2 = np.tan(np.pi * label[1])

        # distances 1 e 2: image borders normalization
        dmin = - MAX_M * MAX_WIDTH
        dmax = MAX_HEIGHT - (MIN_M)*MAX_WIDTH

        d1 = (dmax - dmin)*(label[2] + 1)/2 + dmin
        d2 = (dmax - dmin)*(label[3] + 1)/2 + dmin

        print('d1: ', d1)
        label = [m1, m2, d1, d2]

        # DEPROCESS THE IMAGE
        image = image 
        
        return image, label

    @staticmethod
    def scale_labels(labels):
        m1, m2, b1, b2 = labels
        # note that m = yb - ya / xb - xa
        # where the crop process in y are null and in x this crop
        # it is canceled between xb and xa, so:
        # m_cropped = m
        # and the resize it is a simple multiplication in the denominator and numerator
        # so: m_resized = m_cropped
        # -----------------------
        # now, as y = a*x + b -> b = y - a*x
        # the crop will make: b = b - CROP_FACTOR_X*MAX_WIDTH
        # and the resize will make: b = RESIZE_FACTOR * (b - CROP_FACTOR_X*MAX_WIDTH)
        m1 = m1 
        m2 = m2 
        b1 = RESIZE_FACTOR * (b1 - CROP_FACTOR_X*MAX_WIDTH)
        b2 = RESIZE_FACTOR * (b2 - CROP_FACTOR_X*MAX_WIDTH)

        return [m1, m2, b1, b2]