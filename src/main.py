#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script describe a convolutional network, that will be used to infer the angle and the distance in
the image generated by the lidar data. Note that the label of the images were generated manually with help
of the UI generated by the script lidar_tag.py.  

At the end, plot the result of the cost function with the matplotlib library by number of epochs.

@author: Felipe-Tommaselli
"""

import os
import sys
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime 

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchsummary import summary
from torchvision import transforms
from torchvision import datasets
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from efficientnet_pytorch import EfficientNet
from pytorch_vit import ViT

torch.cuda.empty_cache()

from dataloader import *
sys.path.append('../')
from pre_process import *


class RotatedDataset(Subset):
    ''' this class works like a wrapper for the original dataset, rotating the images'''

    def __init__(self, dataset, angles, rot_type):
        super().__init__(dataset, np.arange(len(dataset)))
        self.angles = angles
        self.rot_type = rot_type
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx) # get the original data

        image = data['image']
        label = data['labels']
        angle = random.choice(self.angles) 
        
        key = len(label)
        if key == 3:
            # we suppose m1 = m2, so we can use the same deprocess
            m1, b1, b2 = label
            m2 = m1
            label = [m1, m2, b1, b2]
        elif key == 4:
            m1, m2, b1, b2 = label
            label = [m1, m2, b1, b2]

        # select the rotation point from the middle or the axis
        if self.rot_type == 'middle': 
            rot_point_np = np.array([112, 112])
            rot_point = (112, 112)
        elif self.rot_type == 'axis':
            rot_point_np = np.array([112, 224])
            rot_point = (112, 224)

        # this only works with PIL, some temporarlly conversion is needed
        pil_image = Image.fromarray(image)

        # Rotate the image around the rotation point with "1D white" background
        rotated_pil_image = transforms.functional.rotate(pil_image, int(angle), fill=255, center=rot_point)

        # convert back to numpy
        rotated_image = np.array(rotated_pil_image)
        
        label =  PreProcess.deprocess(rotated_image, label)
        m1, m2, b1, b2 = label
        x1 = np.arange(0, rotated_image.shape[0])
        x2 = np.arange(0, rotated_image.shape[0])

        y1 = m1*x1 + b1
        y2 = m2*x2 + b2

        # ROTATION MATRIX (for the labels)
        rotation_matrix = np.array([[np.cos(np.radians(angle)), np.sin(np.radians(angle)), rot_point_np[0]*(1-np.cos(np.radians(angle)))-rot_point_np[1]*np.sin(np.radians(angle))], 
                                    [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), rot_point_np[1]*(1-np.cos(np.radians(angle)))+rot_point_np[0]*np.sin(np.radians(angle))], 
                                    [0, 0, 1]])

        # add one dim to the points for matrix multiplication
        points1 = np.stack((x1, y1, np.ones_like(x1)))
        points2 = np.stack((x2, y2, np.ones_like(x2)))

        # apply transformation
        transformed_points1 = rotation_matrix @ points1
        transformed_points2 = rotation_matrix @ points2

        # get the new points
        x1_rotated = transformed_points1[0]
        y1_rotated = transformed_points1[1]
        x2_rotated = transformed_points2[0]
        y2_rotated = transformed_points2[1]

        # get the new line parameters
        m1r, b1r = np.polyfit(x1_rotated, y1_rotated, 1)
        m2r, b2r = np.polyfit(x2_rotated, y2_rotated, 1)

        if key == 3: 
            rotated_label = [m1r, b1r, b2r]
        elif key == 4:
            rotated_label = [m1r, m2r, b1r, b2r]

        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].plot(x1, y1, color='green')
        # ax[0].plot(x2, y2, color='green')
        # ax[0].imshow(image)
        # plt.title(f'nn_cnn: {idx}, angle: {angle}, rot_type: {self.rot_type}')
        # ax[1].plot(x1_rotated, y1_rotated, color='red')
        # ax[1].plot(x2_rotated, y2_rotated, color='red')
        # ax[1].imshow(rotated_image)
        # plt.show()

        # normalize
        rotated_label = NnDataLoader.process_label(rotated_label)

        return {"image": rotated_image, "labels": rotated_label, "angle": angle} 


def transformData(dataset):
    ''' This function garantees that the Data Augmentation occurs!
    I opted for 2 different transformations (both rotations):
        1. Rotation on the middle of the image (112, 112)
        2. Rotation on the axis of the points (112, 224)
        ----------------
        |              |
        |              |
        |     (1)      |
        |              |
        |              |
        ------(2)-------
    
    I select random images from the original dataset, create two new datasets with 
    the rotated images and then concatenate them. With that I can get more images 
    for training and also garantee that the Data Augmentation occurs.
    '''
    
    # both datasets have the same size and dont replace the original images
    num_rotated = int(len(dataset) * 0.2)
    rotated_indices = np.random.choice(len(dataset), num_rotated, replace=False)
    
    # this line create one subset with the indices above for the RotatedDataset class that mount the dataset
    # the lambda function bellow basically remove the 0 angle (no rotation) from the list of angles
    # both the dataset (middle or axis) have the same size but not the same images
    rotated_dataset1 = RotatedDataset(Subset(dataset, rotated_indices), 
                                    angles = np.array(list(filter(lambda x: x != 0, np.arange(-20, 20, 2)))), 
                                    rot_type = 'middle')
    rotated_dataset2 = RotatedDataset(Subset(dataset, rotated_indices), 
                                    angles = np.array(list(filter(lambda x: x != 0, np.arange(-20, 20, 2)))), 
                                    rot_type = 'axis')
    concat_dataset = ConcatDataset([dataset, rotated_dataset1, rotated_dataset2])
    return concat_dataset


def getData(csv_path, train_path, batch_size, num_workers=0):
    ''' get images from the folder (assets/images) and return a DataLoader object '''
    
    dataset = NnDataLoader(csv_path, train_path)

    print(f'dataset size (no augmentation): {len(dataset)}')
    #! artificial não se beneficia muito disso
    # dataset = transformData(dataset)
    print(f'dataset size (w/ augmentation): {len(dataset)}')

    train_size, val_size = int(0.8*len(dataset)), np.ceil(0.2*len(dataset)).astype('int')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_data  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    print(f'train size: {train_size}, val size: {val_size}')
    _ = input('----------------- Press Enter to continue -----------------')
    return train_data, val_data


def cross_validate(model, criterion, optimizer, scheduler, data, num_epochs, num_splits=5):
    #TODO: IMPLEMENT THIS
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    avg_train_losses = []
    avg_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold+1}:")

        train_data = data[train_idx]
        val_data = data[val_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = YourModel()  # Initialize your model here
        optimizer = optim.SGD(model.parameters(), lr=lr)

        results = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs)

        avg_train_losses.append(results['train_losses'])
        avg_val_losses.append(results['val_losses'])

    return avg_train_losses, avg_val_losses


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs):

    train_losses = []
    val_losses = []
    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            images, labels = data['image'], data['labels']
            # convert to float32 and send it to the device
            # image dimension: (batch, channels, height, width)
            images = images.type(torch.float32).to(device)
            images = images.unsqueeze(1)
            # convert labels to float32 and send it to the device
            labels = [label.type(torch.float32).to(device) for label in labels]
            # convert labels to tensor
            labels = torch.stack(labels)
            # convert to format: tensor([[value1, value2, value3, value4], [value1, value2, value3, value4], ...])
            # this is: labels for each image, "batch" times -> shape: (batch, 4)
            labels = labels.permute(1, 0)    
            outputs = model(images)
            loss = criterion(outputs, labels) 
            # Loss_with_L2 = Loss_without_L2 + λ * ||w||^2
            # Calculate L2 regularization loss
            l2_regularization_loss = 0
            for param in model.parameters():
                l2_regularization_loss += torch.norm(param, 2)
            loss += weight_decay * l2_regularization_loss  # Add L2 regularization loss to the total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        else:
        # valing the model
            with torch.no_grad():
                # Set the model to evaluation mode
                model.eval()
                total = 0
                val_loss = 0
                for i, data in enumerate(val_loader):
                    images, labels = data['image'], data['labels']
                    # image dimension: (batch, channels, height, width)
                    images = images.type(torch.float32).to(device)
                    images = images.unsqueeze(1)
                    labels = [label.type(torch.float32).to(device) for label in labels]
                    labels = torch.stack(labels)
                    labels = labels.permute(1, 0)
                    labels_list.append(labels)
                    total += len(labels)
                    outputs = model.forward(images) # propagação para frente
                    # get the predictions to calculate the accuracy
                    _, preds = torch.max(outputs, 1)
                    val_loss += criterion(outputs, labels).item()
                val_losses.append(val_loss/len(val_loader))
            pass
        train_losses.append(running_loss/len(train_loader))
        val_losses.append(running_loss/len(val_loader))
        print(f'[{epoch+1}/{num_epochs}] .. Train Loss: {train_losses[-1]:.5f} .. val Loss: {val_losses[-1]:.5f}')


    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    
    return results

def plotResults(results, epochs, lr):
    # losses
    # print both losses side by side with subplots (1 row, 2 columns)
    # ax1 for train losses and ax2 for val losses
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(results['train_losses'], label='Train Loss')
    ax1.legend(frameon=False)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")

    ax2.plot(results['val_losses'], label='Val Loss')
    ax2.legend(frameon=False)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Val Loss")
    ax2.set_title(f"lr = {lr}")

    # save the plot in the current folder
    day_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") 
    fig.savefig(f'losses_lr={lr}_{day_time}.png')
    
    #plt.show()
    # accuracy
    # plt.plot(results['accuracy_list'], label='Accuracy')
    # plt.legend(frameon=False)
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy")
    # plt.show()


if __name__ == '__main__':
    ############ START ############
    # Set the device to GPU if available
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print('Using {} device'.format(device))

    ############ PARAMETERS ############    
    epochs = 10
    lr = 0.005 # TODO: test different learning rates
    step_size = 15 # TODO: test different step sizes
    gamma = 0.5
    batch_size = 80 # 160 AWS
    weight_decay = 1e-4 # L2 regularization

    ############ DATA ############
    csv_path = "../artificial_data/tags/Artificial_Label_Data4.csv"
    # train_path = os.getcwd() + SLASH + 'artificial_data' + SLASH + 'train4' + SLASH
    train_path = os.path.join(os.getcwd(), '..', 'artificial_data', 'train4')
    train_data, val_data = getData(batch_size=batch_size, csv_path=csv_path, train_path=train_path)

    ############ MODEL ############
    #?model = models.resnet18(pretrained=True)
    #?model = models.resnet50(pretrained=True)
    #?model = EfficientNet.from_pretrained('efficientnet-b0')
    #?model = models.vgg16(pretrained=True)
    #?model = models.mobilenet_v2(pretrained=True)


    ########### ViT NET ###########
    # Load the ViT model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=3,
        dim=512,          # Adjust based on your requirements
        depth=6,
        heads=8,
        mlp_dim=512
    )

    ########### EFFICENT NET ###########
    # model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # num_ftrs = model._fc.in_features
    # model._fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(512, 256),
    #     nn.BatchNorm1d(256),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(256, 3)  
    # )

    ########### VGG NET 16 ########### 
    # model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    # # Accessing the classifier part of the VGG16 model
    # num_ftrs = model.classifier[6].in_features
    # model.classifier[6] = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(512, 256),
    #     nn.BatchNorm1d(256),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(256, 3)
    # )

    ########### MOBILE NET ########### 
    # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # # MobileNetV2 uses a different attribute for the classifier
    # num_ftrs = model.classifier[1].in_features
    # model.classifier[1] = nn.Sequential(
    # nn.Linear(num_ftrs, 512),
    # nn.BatchNorm1d(512),
    # nn.ReLU(inplace=True),
    # nn.Linear(512, 256),
    # nn.BatchNorm1d(256),
    # nn.ReLU(inplace=True),
    # nn.Linear(256, 3)
    # )

    #################################

    # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # num_ftrs = model.fc.in_features
    # # Adding batch normalization and an additional convolutional layer
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.BatchNorm1d(512),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(512, 256),
    #     nn.BatchNorm1d(256),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(256, 3)
    # )

    # Moving the model to the device (GPU/CPU)
    model = model.to(device)
    ############ NETWORK ############
    criterion = nn.L1Loss() # TODO: test different loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    ############ DEBBUG ############
    #summary(model, (1, 224, 224))
    #print(model)

    ############ TRAINING ############
    results = train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_loader=train_data, val_loader=val_data, num_epochs=epochs)

    ############ RESULTS ############
    plotResults(results, epochs, lr)

    ############ SAVE MODEL ############
    # get day for the name of the file
    day_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") 
    path = os.getcwd() + '/models/' + 'model' + '_' + str(lr).split('.')[1] + '_' + str(day_time) + '.pth'
    torch.save(model.state_dict(), path)
    print(f'Saved PyTorch Model State to:\n{path}')
