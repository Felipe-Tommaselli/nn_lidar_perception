# -*- coding: utf-8 -*-
"""


@author: andres
@author: Felipe-Tommaselli
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import os

#filedata = open('/home/andres/Documents/learning_lidar/syncro_data.csv',"r")
filedata = open('syncro_data_validation.csv',"r")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data_validation.csv',"w")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data.csv',"w")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data_norm.csv',"w")
#finalData = open('/home/andres/Documents/learning_lidar/filter_syncro_data_validation_norm.csv',"w")
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