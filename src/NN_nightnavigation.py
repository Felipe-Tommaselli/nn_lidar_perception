#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: andres
@author: Felipe-Tommaselli
"""

import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
#from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split

class LidarDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None):
        self.labels = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dl = self.labels.iloc[idx, 2]
        dr = self.labels.iloc[idx, 3]
        dratio = self.labels.iloc[idx, 4]
        heading = self.labels.iloc[idx, 1]
        lidar = np.empty(1081)
        for step in range(0,len(self.labels.iloc[idx,7:])):
            lidar[step] = (self.labels.iloc[idx, step+7])
        torch.from_numpy(lidar)
        sample = {"dl": dl , "dr": dr, "dratio":dratio, "heading": heading, "lidar": lidar}
        #print(sample)
        return sample

def fetch_dataloader(data_dir, batch_size,num_workers):
    dl = DataLoader(LidarDataset(data_dir), batch_size=batch_size, shuffle=True,num_workers=num_workers)
    #print(dl)
    dataloaders = dl
    return dataloaders


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, hidden_size2, hidden_size3, hidden_size4, hidden_size5, outputs):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.hidden_size2 = hidden_size2
            self.hidden_size3 = hidden_size3
            self.hidden_size4 = hidden_size4
            self.hidden_size5 = hidden_size5
            #self.hidden_size6 = hidden_size6
            self.outputs = outputs
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size2)
            self.fc4 = torch.nn.Linear(self.hidden_size2, self.hidden_size3)
            self.fc5 = torch.nn.Linear(self.hidden_size3, self.hidden_size4)
            #self.fc6 = torch.nn.Linear(self.hidden_size4, self.hidden_size5)
            self.fc7 = torch.nn.Linear(self.hidden_size4, self.hidden_size5)            
            self.fc2 = torch.nn.Linear(self.hidden_size5, self.outputs)
            self.dpout1 = torch.nn.Dropout(p=0.3)
            self.dpout2 = torch.nn.Dropout(p=0.5)
            self.dpout3 = torch.nn.Dropout(p=0.5)            
            #self.fc2 = torch.nn.Linear(self.hidden_size, 2)
                   
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            dp1 = self.dpout1(relu)
            hidden2 = self.fc3(dp1)
            relu2 = self.relu(hidden2)
            dp3 = self.dpout1(relu2)
            hidden3 = self.fc4(dp3)
            relu3 = self.relu(hidden3)
            hidden4 = self.fc5(relu3)
            relu4 = self.relu(hidden4)
            dp2 = self.dpout2(relu4)
            #hidden5 = self.fc6(dp2)
            #relu5 = self.relu(hidden5)
            hidden6 = self.fc7(dp2)
            relu6 = self.relu(hidden6)
            output = self.fc2(relu6)
            return output

def TrainCollection():
    collection_name = '/home/andres/Documents/learning_lidar/filter_syncro_data_norm.csv'
    return collection_name

def ValCollection():
    collection_name = '/home/andres/Documents/learning_lidar/filter_syncro_data_validation_norm.csv'
    return collection_name

def trainFunction(nc,batch_size):
    train_loss =[]
    train_heading_lossL1 =[]
    #train_ratio_lossL1 =[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for step in range (1,nc):
            name = TrainCollection()
            path = name
            dl = fetch_dataloader(path,batch_size,num_workers=0)    
            print(path)
            #print('epoch:', step)
            for batch in dl:    
                batch_heading = batch['heading']
                batch_heading_float = batch_heading.float()
                batch_heading_gpu = batch_heading_float.to(device)
                #batch_ratio = batch['dratio'] 
                #batch_ratio_float = batch_ratio.float()
                #batch_ratio_gpu = batch_ratio_float.to(device)
                batch_lidar = batch['lidar']        
                batch_lidar_float = batch_lidar.float()
                batch_lidar_gpu = batch_lidar_float.to(device)       
                pred = net(batch_lidar_gpu)
                net.to(device)
                heading_loss = mse_loss(pred[:,0],batch_heading_gpu)
                #ratio_loss = mse_loss(pred[:,1],batch_ratio_gpu)
                heading_lossL1 = L1_loss(pred[:,0],batch_heading_gpu)
                #ratio_lossL1 = L1_loss(pred[:,1],batch_ratio_gpu)
                output = heading_loss
                optimizer.zero_grad()
                output.backward()
                optimizer.step()
                #print('training_ratio: ',ratio_lossL1,'training_heading: ',heading_lossL1)
                print('training_heading: ',heading_lossL1)
                train_loss.append(output.item())
                train_heading_lossL1.append(heading_lossL1.item())
                #train_ratio_lossL1.append(ratio_lossL1.item())
    return train_loss, train_heading_lossL1#, train_ratio_lossL1

def ValidFunction(nc, batch_size):
    Val_loss =[] 
    Val_heading_lossL1 =[] 
    #Val_ratio_lossL1 =[] 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for step in range (1,nc):
            name = ValCollection()
            path = name
            dl = fetch_dataloader(path,batch_size,num_workers=0)    
            print(path)
            #print('epoch:', step)
            for batch in dl:    
                batch_heading = batch['heading']
                batch_heading_float = batch_heading.float()
                batch_heading_gpu = batch_heading_float.to(device)
                #batch_ratio = batch['dratio'] 
                #batch_ratio_float = batch_ratio.float()
                #batch_ratio_gpu = batch_ratio_float.to(device)
                batch_lidar = batch['lidar']        
                batch_lidar_float = batch_lidar.float()
                batch_lidar_gpu = batch_lidar_float.to(device)
                Val = net(batch_lidar_gpu)
                net.to(device)
                heading_loss = mse_loss(Val[:,0],batch_heading_gpu)
                #ratio_loss = mse_loss(Val[:,1],batch_ratio_gpu)
                heading_lossL1 = L1_loss(Val[:,0],batch_heading_gpu)
                #ratio_lossL1 = L1_loss(Val[:,1],batch_ratio_gpu)
                output = heading_loss # + ratio_loss
                Val_loss.append(output.item())  
                Val_heading_lossL1.append(heading_lossL1 .item())
                #Val_ratio_lossL1.append(ratio_lossL1 .item())
                #print('Val_ratio: ',ratio_lossL1,'Val_heading: ',heading_lossL1)
                print('Val_heading: ',heading_lossL1)
    return Val_loss, Val_heading_lossL1 #, Val_ratio_lossL1
    
if __name__ == '__main__':
    # check if cuda is available
    print(torch.cuda.is_available())

    
    #batch_size = 4
    epoch_number = 100
    mean_train = []
    mean_val = []
    mean_trainL1 = []
    mean_train_headingL1 = []
    #mean_train_ratioL1 =[]
    mean_val_headingL1 = []
    #mean_val_ratioL1 = []
    #Call Model of NN
    net = Feedforward(1081,3500,1800,1200,600,200, 1)#input, hidden1, hidden2, hidden3, outputs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    #define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    #Define loss function
    mse_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()     
    init_val = 1
    for epoch in range(0,epoch_number):
        net.train()
        print('epoch:', epoch)
        #Train_Loss, Train_heading_LossL1, Train_ratio_LossL1 = trainFunction(2,50)#maximal is 16
        Train_Loss, Train_heading_LossL1 = trainFunction(2,50)#maximal is 16
        mean_train.append(mean(Train_Loss))
        mean_train_headingL1.append(mean(Train_heading_LossL1))
        #mean_train_ratioL1.append(mean(Train_ratio_LossL1))
        net.eval()
        #Eval_Loss, Eval_heading_LossL1, Eval_ratio_LossL1 = ValidFunction(2,50)#maximal is 6
        Eval_Loss, Eval_heading_LossL1 = ValidFunction(2,50)#maximal is 6
        mean_val.append(mean(Eval_Loss))
        mean_val_headingL1.append(mean(Eval_heading_LossL1))
        print(mean(Eval_heading_LossL1))
        
        #mean_val_ratioL1.append(mean(Eval_ratio_LossL1))
        if epoch == (init_val*5)-1:
            init_val = init_val + 1
            plt.figure(1)
            plt.plot(mean_train,'r')   
            plt.plot(mean_val,'b')   
            plt.figure(2)
            plt.plot(mean_train_headingL1,'r')   
            plt.plot(mean_val_headingL1,'b')
            #plt.figure(3)
            #plt.plot(mean_train_ratioL1,'r')   
            #plt.plot(mean_val_ratioL1,'b')
            plt.show()
            
    
    
    