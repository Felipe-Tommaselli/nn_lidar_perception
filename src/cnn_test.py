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
from nn_cnn import ResidualBlock, NetworkCNN

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getData(csv_path, batch_size=7, num_workers=0):
    ''' get images from the folder (assets/images) and return a DataLoader object '''
    
    dataset = LidarDatasetCNN(csv_path, train=True)

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
train_data, val_data = getData(csv_path="~/Documents/IC_NN_Lidar/assets/tags/Label_Data.csv")

# test the model with the validation data for one random image
# showing the image and the predicted and real labels
# get the first image from the validation data
for i, data in enumerate(train_data):
    images = data['image']
    labels = data['labels']
    print('l:', labels)
    break

# image dimension: batch x 1 x 650 x 650 (batch, channels, height, width)
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
model.load_state_dict(torch.load(os.getcwd() + '/model.pth'))
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

x = np.linspace(0, 507, 507) # 0 to 507 with 507 points
p1 = predictions[0][0] * x + predictions[0][2]
p2 = predictions[0][1] * x + predictions[0][3]

l1 = label[0][0] * x + label[0][1]
l2 = label[0][2] * x + label[0][3]

plt.plot(x, p1, '-r', label=f'p1={predictions[0][0]}x+{predictions[0][2]}', linewidth=2)
plt.plot(x, p2, 'r', label=f'p2={predictions[0][1]}x+{predictions[0][3]}', linewidth=2)
plt.plot(x, l1, '-b', label=f'l1={label[0][0]}x+{label[0][2]}', linewidth=2)
plt.plot(x, l2, 'b', label=f'l2={label[0][1]}x+{label[0][3]}', linewidth=2)
# show the image
plt.imshow(image, cmap='gray')
plt.show()