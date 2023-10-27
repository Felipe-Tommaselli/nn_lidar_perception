import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
from sklearn.metrics import roc_curve, auc

torch.cuda.empty_cache()

from artificial_dataloader import *
sys.path.append('../')
from pre_process import *

# set the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def getData(csv_path, batch_size=7, num_workers=0):
    ''' get images from the folder (assets/images) and return a DataLoader object '''
    
    dataset = ArtificialLidarDatasetCNN(csv_path)

    train_size, val_size = int(0.8*len(dataset)), np.ceil(0.2*len(dataset)).astype('int')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_data  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    # get one image shape from the train_data
    for i, data in enumerate(train_data):
        print(f'images shape: {data["image"].shape}')
        print(f'label length: {len(data["labels"])}')
        break
    print('-'*65)

    print(f'train size: {train_size}, val size: {val_size}')
    _ = input('----------------- Press Enter to continue -----------------')
    return train_data, val_data

# Get the data
csv_path = "../../artificial_data/tags/Artificial_Label_Data3.csv"
train_data, val_data = getData(csv_path=csv_path)

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
############ MODEL ############
# load the model from the saved file
# model = NetworkCNN(ResidualBlock).to(device)
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = model.fc.in_features
# Adding batch normalization and an additional convolutional layer
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 3)  # Alterado para 3 valores de saída
)

# Moving the model to the device (GPU/CPU)
model = model.to(device)
path = os.getcwd() + '/models/' + '/model_01_26-10-2023.pth'
model.load_state_dict(torch.load(path))
model.eval()

# image it is the first image from the images batch
image = images[0].unsqueeze(0)
# label it is the first label from the labels batch
label = labels[0].unsqueeze(0)

# Inicie a contagem de tempo antes da inferência
start_time = time.time()

# get the model predictions
predictions = model(image)

# Encerre a contagem de tempo após a inferência
end_time = time.time()

# convert the predictions to numpy array
predictions = predictions.to('cpu').cpu().detach().numpy()
# convert the labels to numpy array
label = labels.to('cpu').cpu().detach().numpy()

print('Inference time: {:.4f} ms'.format((end_time - start_time)*1000))

# print the predictions and labels
print('predictions:', predictions)
print('label:', label[0])

# convert image to cpu 
image = image.to('cpu').cpu().detach().numpy()
# image it is shape (1, 1, 507, 507), we need to remove the first dimension
image = image[0][0]

print('>>> image shape:', image.shape)
print('>>> label :', label[0])
print('>>> predictions:', predictions[0])

label = PreProcess.deprocess(image=image, label=label[0].tolist())
predictions = PreProcess.deprocess(image=image, label=predictions[0].tolist())

print('label (deprocessed):', label)
print('predictions (deprocessed):', predictions)

# plot the labels and the predictions on the image
# note that labels and predicts are an array of 4 values
# [m1, m2, b1, b2] -> m1, m2 are the slopes and b1, b2 are the intercepts of 2 lines
# the first line is the left line and the second line is the right line
# the lines are the borders of the road
# the image is 540x540 pixels

# plot
fig, ax = plt.subplots()

# get the slopes and intercepts
m1p, m2p, b1p, b2p = predictions

# get the x and y coordinates of the lines
x1 = np.arange(0, 224)
y1p = m1p*x1 + b1p
x2 = np.arange(0, 224)
y2p = m2p*x2 + b2p


# add text labels
y_value = 50 # fix the y value
# calculate the corresponding x values using the inverse equation of the line
x1_value = (y_value - b1p) / m1p
x2_value = (y_value - b2p) / m2p

# get the slopes and intercepts
m1, m2, b1, b2 = label
m2 = m1
# get the x and y coordinates of the lines
x1 = np.arange(0, 224)
y1 = m1*x1 + b1
x2 = np.arange(0, 224)
y2 = m2*x2 + b2

# plot the lines
ax.plot(x1, y1, color='red')
ax.plot(x2, y2, color='red')

y1t = (m1 + 0.3)*(x1-4) + b1 - 2
y2t = (m2 + 0.5)*(x2) + b2 + 2

ax.plot(x1p, y1p, color='blue')
ax.plot(x2p, y2p, color='blue')

# legend in upper right: real and predicted. real in red and predicted in blue
plt.annotate('Real ---',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            xytext=(10, -10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='white', lw=0.1),
            color='red')

plt.annotate('Predicted ---',
            xy=(0.05, 0.90),
            xycoords='axes fraction',
            xytext=(10, -10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='white', lw=0.1),
            color='blue')
# show the image
ax.imshow(image, cmap='gray')
plt.show()
