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

global RAW_SIZE
global MAX_M

global CROP_FACTOR_X
global DESIRED_SIZE
global RESIZE_FACTOR

RAW_SIZE = 540
MAX_M = 540 
CROP_FACTOR_X = 0.17 #% # using this for square image assure
# AlexNet famous input size (224x224 pxs)
DESIRED_SIZE = 224 #px

class PreProcess:

    def __init__(self, dataset) -> None:
        ''' Constructor of the class. '''

        self.labels = copy.deepcopy(dataset['labels'])
        self.image = copy.deepcopy(dataset['image'])
        
        self.resize_factor = DESIRED_SIZE / self.image.shape[0] # FOR NOW 
        self.cropped_size = self.image.shape[0] # FOR NOW

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
        # x 0,0 
        # ----------------
        # |              |
        # | ------------ |
        # | |          | |
        # | |          | |
        # | |          | |
        # | |          | |
        # ----------------
        # double crop in x from each side (cropped_x is the lost region)
        # one big crop in y from the top (cropped_y is the remaining region)
        
        # correcting in y for the image continue to be a square
        cropped_size_x = int(MAX_WIDTH * CROP_FACTOR_X)
        cropped_size_y = int(MAX_WIDTH - 2 * cropped_size_x) 

        cropped_image = image[MAX_HEIGHT - cropped_size_y: MAX_HEIGHT, cropped_size_x:MAX_WIDTH-cropped_size_x]
        
        resized_image = cv2.resize(cropped_image, (DESIRED_SIZE, DESIRED_SIZE))

        #* storage this for later
        self.resize_factor = DESIRED_SIZE/ int(cropped_image.shape[0])
        self.cropped_size = int(cropped_image.shape[0])

        return resized_image

    def process_label(self, labels: list) -> list:
        ''' Returns the processed label. '''

        # scale the labels acording to the image size
        labels = PreProcess.scale_labels(self.cropped_size, labels, self.resize_factor)

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
        # note that for this step we are already working with the cropped image
        # so we will use DESRIED_SIZE
        # and, the MAX_M for square cases is = DESIRED_SIZE
        # -------------------------
        # where the minimum distance is when:
        # y = 0 and x = MAX_WIDTH with m = MAX_M
        # so b = 0 - (MAX_M)*MAX_WIDTH <- minimum distance
        MAX_M = DESIRED_SIZE
        dmin = - MAX_M * DESIRED_SIZE
        # and the maximum distance is when:
        # y = MAX_HEIGTH and x = MAX_WIDTH with m = MIN_M
        # so b = MAX_HEIGHT - (MIN_M)*MAX_WIDTH <- maximum distance
        dmax = DESIRED_SIZE - (-MAX_M)*DESIRED_SIZE

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
        MAX_M = DESIRED_SIZE
        dmin = - MAX_M * DESIRED_SIZE
        dmax = DESIRED_SIZE - (-MAX_M)*DESIRED_SIZE

        d1 = (dmax - dmin)*((label[2] + 1)/2) + dmin
        d2 = (dmax - dmin)*((label[3] + 1)/2) + dmin

        label = [m1, m2, d1, d2]

        return label

    @staticmethod
    def scale_labels(image_size, labels, resize_factor):

        # correcting the y crop (without the resize)
        # the image_size without the resize it is the cropped_size
        CROP_FACTOR_Y = image_size / RAW_SIZE 

        m1, m2, b1, b2 = labels
        # note that m = yb - ya / xb - xa
        # where the crop process in y are null and in x this crop
        # it is canceled between xb and xa, so:
        # m_cropped = m
        # and the resize it is a simple multiplication in the denominator and numerator
        # so: m_resized = m_cropped
        # -----------------------
        # now, as y = a*x + b -> b = y - a*x
        # the crop will make: b = b - m1*CROP_FACTOR_X*MAX_WIDTH + (1 - CROP_FACTOR_Y)*MAX_HEIGHT
        # and the resize will make: b = RESIZE_FACTOR * (b - ...)
        # -----------------------
        # note that this come from the expression y' = y - CROP_FACTOR_Y*MAX_HEIGHT and with 
        # same for x and x', we can see that the new y and x make this transformations
        # DONT FORGET M1 IN b1 CORRECTION (it took me some hours to debbug that)
        m1 = m1 
        m2 = m2 

        b1 = resize_factor * (b1 + m1*CROP_FACTOR_X*RAW_SIZE - (1 - CROP_FACTOR_Y)*RAW_SIZE)
        b2 = resize_factor * (b2 + m2*CROP_FACTOR_X*RAW_SIZE - (1 - CROP_FACTOR_Y)*RAW_SIZE)

        return [m1, m2, b1, b2]