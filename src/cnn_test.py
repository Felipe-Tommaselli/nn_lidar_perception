import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary

torch.cuda.empty_cache()

from dataloader import *
from pre_process import *

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getData(csv_path, batch_size=7, num_workers=0):
    ''' get images from the folder (assets/images) and return a DataLoader object '''
    
    dataset = LidarDatasetCNN(csv_path)

    train_size, val_size = int(0.8*len(dataset)), np.ceil(0.2*len(dataset)).astype('int')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_data  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    # get one image shape from the train_data
    for i, data in enumerate(train_data):
        print(f'images shape: {data["image"].shape}')
        break
    print('-'*65)
    return train_data, val_data

# Get the data
train_data, val_data = getData(csv_path="/home/tommaselli/Documents/IC_NN_Lidar/data/tags/Label_Data1.csv")

# test the model with the validation data for one random image
# showing the image and the predicted and real labels
# get the first image from the validation data
for i, data in enumerate(train_data):
    images = data['image']
    labels = data['labels']
    print('l:', labels)
    break

# image dimension: (batch, channels, height, width)
images = images.type(torch.float32).to(device)
images = images.unsqueeze(1)

labels = [label.type(torch.float32).to(device) for label in labels]
# convert labels to tensor
labels = torch.stack(labels)
# convert to format: tensor([[value1, value2, value3, value4], [value1, value2, value3, value4], ...])
# this is: labels for each image, "batch" times -> shape: (batch, 4)
labels = labels.permute(1, 0)

# load the model from the saved file
model = NetworkCNN(ResidualBlock)
model = model.to(device)
model.load_state_dict(torch.load(os.getcwd() + '/model0.0113-06-2023.pth'))
model.eval()

# image it is the first image from the images batch
image = images[0].unsqueeze(0)
# label it is the first label from the labels batch
label = labels[0].unsqueeze(0)

# get the model predictions
predictions = model(image)
# convert the predictions to numpy array
predictions = predictions.to('cpu').cpu().detach().numpy()
# convert the labels to numpy array
label = labels.to('cpu').cpu().detach().numpy()

# print the predictions and labels
print('predictions:', predictions)
print('label:', label[0])

# convert image to cpu 
image = image.to('cpu').cpu().detach().numpy()
# image it is shape (1, 1, 507, 507), we need to remove the first dimension
image = image[0][0]

image, label = PreProcess.deprocess(image, label[0])
#TODO: deporcess predictions

print('label (deprocessed):', label)

# plot the labels and the predictions on the image
# note that labels and predicts are an array of 4 values
# [m1, m2, b1, b2] -> m1, m2 are the slopes and b1, b2 are the intercepts of 2 lines
# the first line is the left line and the second line is the right line
# the lines are the borders of the road
# the image is 540x540 pixels

# get the slopes and intercepts
m1, m2, b1, b2 = predictions[0]
# get the x and y coordinates of the lines
x1 = np.arange(0, 224)
y1 = m1*x1 + b1
x2 = np.arange(0, 224)
y2 = m2*x2 + b2

# plot the lines
plt.plot(x1, y1, color='red')
plt.plot(x2, y2, color='red')

# get the slopes and intercepts
m1, m2, b1, b2 = label
# get the x and y coordinates of the lines
x1 = np.arange(0, 224)
y1 = m1*x1 + b1
x2 = np.arange(0, 224)
y2 = m2*x2 + b2

# plot the lines
plt.plot(x1, y1, color='green')
plt.plot(x2, y2, color='green')

# show the image
plt.imshow(image, cmap='gray')
plt.show()