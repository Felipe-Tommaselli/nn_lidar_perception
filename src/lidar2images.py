# -*- coding: utf-8 -*-
"""
lidar2images.py is a script that converts lidar data to images.

With the lidar data obtained from the syncro_data.csv file, witch contains data from the TerraSentia lidar, this script convert 
the data to images and save them in the images folder. The image format is basically a plot of the lidar data in a 2D space.

The script is divided in 3 parts:
    1. Data processing
    2. Data visualization
    3. Data saving

The data processing part is divided in 3 functions:
    1. limitLidar: This function limits the lidar data to a maximum value of 5 meters.
    2. polar2xy: This function converts the polar coordinates of the lidar data to cartesian coordinates.
    3. plot_lines: This function plots the lidar data in a 2D space.

The data visualization part is divided in 2 functions:
    1. plot_lines: This function plots the lidar data in a 2D space.

The data saving part only contains the function of saving the images.

The script is executed by running the following command in the terminal:
> python lidar2images.py


@author: andres
@author: Felipe-Tommaselli
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import os

# move from root (\src) to \assets\tags
os.chdir('..') 
path = os.getcwd() + '\\datasets\\'
data_file_name = os.path.join(path,'syncro_data_validation.csv')

#filedata = open('/home/andres/Documents/learning_lidar/syncro_data.csv',"r")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data_validation.csv',"w")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data.csv',"w")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data_norm.csv',"w")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data_validation_norm.csv',"w")
filedata = open(data_file_name,"r")
data = filedata.readlines()

def limitLidar(readings):
    final_readings= []
    for t in range(0,len(readings)):
            final_readings.append(int(readings[t])/1000)
    #print(readings)
    return final_readings #, maxval

def polar2xy(lidar):
        x_lidar = []
        y_lidar = []
        min_angle = np.deg2rad(-45)
        max_angle = np.deg2rad(225)
        angle = np.linspace(min_angle, max_angle, 1081, endpoint = False)
        for i in range(0, len(lidar)):
            x_lidar.append(lidar[i]*math.cos(angle[i]))
            y_lidar.append(lidar[i]*math.sin(angle[i]))
            
        return x_lidar, y_lidar

def plot_lines(xl,yl,t):
    LW=0.8
    plt.cla()
    plt.plot(xl,yl,'g.')
    plt.xlim(-LW,LW)
    plt.ylim(-1,5)
    plt.axis('off')
    plt.pause(0.1)
    print(t)
    # name = '/home/andres/Documents/learning_lidar/images_approach/images/image'+str(t)
    #plt.savefig(name)
    plt.savefig(os.getcwd() + '\\assets\\images' + '\\image' + str(t))


for step in range(0,len(data)):
    lidar = (data[step].split(","))[7:1088]
    #print(lidar)
    lidar_readings = limitLidar(lidar)
    #print(lidar_readings)
    x,y = polar2xy(lidar_readings) 
    #print(x)
    plot_lines(x,y,step)

filedata.close()