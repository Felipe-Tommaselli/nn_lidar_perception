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
import os

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

    #* NON ARTIFICIAL DATA PREPROCESSING
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

        contoured_image = PreProcess.contours_image(resized_image)

        # normalize the image (0 to 255) -> (0 to 1)
        # resized_image = resized_image / 255 #! does not work well 

        return contoured_image

    #* NON ARTIFICIAL DATA PREPROCESSING
    def process_label(self, labels: list) -> list:
        ''' Returns the processed label. '''

        # scale the labels acording to the image size
        labels = PreProcess.scale_labels(self.cropped_size, labels, self.resize_factor)

        # NORMALIZE THE AZIMUTH 1 AND 2 
        m1 = labels[0]
        m2 = labels[1]

        return [m1, m2, d1, d2]

    #* NON ARTIFICIAL DATA PREPROCESSING
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

    # ############################################################################################
    #              UTILITIES FUNCTIONS FOR DEPROCESSING AND ROUTINE OPERATIONS
    # ############################################################################################

    @staticmethod
    def deprocess(image, label):
        ''' Returns the deprocessed image and label. '''

        if len(label) == 3:
            # we suppose m1 = m2, so we can use the same deprocess
            print('supposing m1 = m2')   
            w1, q1, q2 = label
            w2 = w1
        elif len(label) == 4:
            print('not supposing m1 = m2')        
            w1, w2, q1, q2 = label

        # DEPROCESS THE LABEL
        q1_original = ((q1 + 1) * (187.15 - (-56.06)) / 2) + (-56.06)
        q2_original = ((q2 + 1) * (299.99 - 36.81) / 2) + 36.81
        w1_original = ((w1 + 1) * (0.58 - (-0.58)) / 2) + (-0.58)
        w2_original = ((w2 + 1) * (0.58 - (-0.58)) / 2) + (-0.58)


        # print(f'labels w1={w1}, w2={w2}, q1={q1}, q2={q2}')
        m1, m2, b1, b2 = PreProcess.parametrization(w1_original, w2_original, q1_original, q2_original)

        label = [m1, m2, b1, b2]

        return label


    @staticmethod
    def extract_label(labels):
        ''' This function aims to extract infos more relevants for the neural network. 
        For now, the only thing that gave performance to the cnn was "b" intersection in 
        a differente parametrization. '''

        m1, m2, b1, b2 = labels

        # imagine: y = m*x + b, but we dont want b that crosses the x = 0 line
        # why? because he can be -291600 to 292140, with that, the network will have to
        # learn a lot of things that are not relevant for the problem. 
        # we can parametrize the line as: y = m*x + b, but we want b that crosses
        # the y = 0 line. For that we can simply change the parametrization to:
        # x = w*y + q (where w = 1/m and q = -b/m). For now, it is better 
        w1, w2, q1, q2 = PreProcess.parametrization(m1, m2, b1, b2)

        # Normalization with empirical values from parametrization.ipynb
        # X_normalized = 2 * (X - MIN) / (MAX - MIN) - 1
        '''
        w1: -0.58 ~ 0.58
        w2: -0.58 ~ 0.58
        q1: -56.06 ~ 187.15
        q2: 36.81 ~ 299.99
        '''        
        q1 = 2*((q1 - (-56.06)) / (187.15 - (-56.06))) - 1
        q2 = 2*((q2 - 36.81)) / ((299.99 - 36.81)) - 1
        w1 = 2*((w1 - (-0.58)) / ((0.58 - (-0.58)))) - 1
        w2 = 2*((w2 - (-0.58)) / ((0.58 - (-0.58)))) - 1

        return [w1, w2, q1, q2]

    @staticmethod
    def parametrization(m1, m2, b1, b2):
        w1 = 1/m1
        w2 = 1/m2
        q1 = -b1/m1
        q2 = -b2/m2
        # note that the in (process) and the out (deprocess) are the same operations
        # we are using the same operations for process and deprocess :)
        return [w1, w2, q1, q2]


    # ############################################################################################
    #         TODO: ADD COMPUTER VISION SEGMENTATION IN A STABLE NN VERSION
    # ############################################################################################


    @staticmethod
    def contours_image(image):
        ''' Returns the image with the contours. '''

        # inverter a imagem 
        img = cv2.bitwise_not(image)

        #* Aplicar um filtro de suavização na imagem para remover ruído
            #* Gaussian Blur: Espalha mais e perde a informaçõa do ponto 
            #* Bilateral Filter: Espalha menos e mantém a informação do ponto
            #* Blur: Espalha menos e pondera a informação (efeito de chacoalho)
        
        ####### DILATAÇÃO ########
        img_blur = cv2.blur(img, (15, 15))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)) # kernel retangular
        img_dilated = cv2.dilate(img_blur, kernel, iterations=1)

        ####### MASCARA CIRCULAR ########
        mask = np.zeros((224, 224), dtype=np.uint8)
        # Criar a máscara circular
        cv2.ellipse(mask, center=(112, 157), axes=(85, 52), angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

        ####### SUAVIZAÇÃO ########
        # Aplicar o desfoque mais leve na região circular
        img_blur_circle = cv2.GaussianBlur(img_dilated, (41, 41), sigmaX=0, sigmaY=20)
        img_blur_circle_masked = cv2.bitwise_and(img_blur_circle, img_blur_circle, mask=mask)
        # Aplicar o desfoque mais forte na imagem original
        img_blur = cv2.GaussianBlur(img_dilated, (61, 61), sigmaX=0, sigmaY=40, borderType=cv2.BORDER_DEFAULT)
        # Subtrair a região circular suavizada da imagem original suavizada
        img_blur_masked = cv2.bitwise_and(img_blur, img_blur, mask=cv2.bitwise_not(mask))
        img_blur_final = cv2.bitwise_or(img_blur_masked, img_blur_circle_masked)
        # blur leve na imagem final 
        # img_blur_final = cv2.GaussianBlur(img_blur_final, (11, 11), sigmaX=0, sigmaY=10)

        ####### EROSÃO ########
        # Erosão fora do circulo (mais forte)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 34))
        img_eroded = cv2.erode(img_blur_final, kernel_erode, iterations=1)
        # blur leve na imagem final 
        img_eroded = cv2.GaussianBlur(img_eroded, (11, 11), sigmaX=0, sigmaY=0)

        ####### BINARIZAÇÃO ########
        # Realizar uma binarização na imagem
        ret, thresh = cv2.threshold(img_eroded, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        ####### CONTORNOS ########
        # Encontrar todos os contornos presentes na imagem
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = np.zeros_like(img)
        cv2.drawContours(contours_img, contours, -1, (255, 255, 255), 2)

        ####### SELECIONAR O MAIOR CONTORNO ########
        # Selecionar o contorno de maior área
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_img = np.zeros_like(img)
        cv2.drawContours(largest_contour_img, [largest_contour], -1, (255, 255, 255), 2)

        ####### DESENHAR O CONTORNO NA IMAGEM ########
        # Desenhar o contorno na imagem com blur
        contourned_img_blur = cv2.drawContours(cv2.bitwise_not(img_eroded), [largest_contour], -1, (0, 255, 0), 3)

        ####### DESENHAR O CONTORNO NA IMAGEM ORIGINAL ########
        # Desenhar o contorno selecionado na imagem original
        contourned_img = cv2.drawContours(cv2.bitwise_not(img), contours, -1, (0, 255, 0), 3)

        ####### MOSTRAR AS IMAGENS ########
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        axs[0, 0].imshow(cv2.bitwise_not(img), cmap='gray')
        axs[0, 0].set_title('Imagem original')
        axs[0, 1].imshow(cv2.bitwise_not(img_dilated), cmap='gray')
        axs[0, 1].set_title('Imagem Dilatada')
        axs[0, 2].imshow(cv2.bitwise_not(img_blur_final), cmap='gray')
        axs[0, 2].set_title('Imagem suavizada')
        axs[0, 3].imshow(cv2.bitwise_not(img_eroded), cmap='gray')
        axs[0, 3].set_title('Imagem Erodida')
        axs[0, 4].imshow(cv2.bitwise_not(thresh), cmap='gray')
        axs[0, 4].set_title('Imagem binarizada')
        axs[1, 0].imshow(cv2.bitwise_not(thresh), cmap='gray')
        axs[1, 0].set_title('Imagem binarizada')
        axs[1, 1].imshow(contours_img, cmap='gray')
        axs[1, 1].set_title('Contornos')
        axs[1, 2].imshow(largest_contour_img, cmap='gray')
        axs[1, 2].set_title('Contorno de maior área')
        axs[1, 3].imshow(contourned_img_blur, cmap='gray')
        axs[1, 3].set_title('Contorno desenhado')
        axs[1, 4].imshow(contourned_img, cmap='gray')
        axs[1, 4].set_title('Contorno desenhado')

        # Save the contours_img in the assets/vision folder
        file_path = os.path.join(folder_path, f"contours_img{np.randint(1000)}.png")
        cv2.imwrite('../assets/vision/', contours_img)

        #plt.show()
        plt.cla()
        plt.close()

        # criar a máscara das bordas internas
        border_size = 6
        mask = np.ones_like(contours_img)
        mask[border_size:-border_size, border_size:-border_size] = 0
        # aplicar a máscara na imagem para definir as bordas internas como preto
        contours_img[mask==1] = 0
        # convert to np array
        contours_img = np.array(contours_img)

        return contours_img
