# -*- coding: utf-8 -*-
"""
lidar2images.py is a script that converts lidar data to images.

With the lidar data obtained from the syncro_data.csv file, witch contains data from the TerraSentia lidar, this script convert 
the data to images and save them in the images folder. The image format is basically a plot of the lidar data in a 2D space.

The script is divided in 3 parts:
    1. Data acquisition
    2. Data processing
    3. Data visualization

The data acquisition part is divided in 1 function:
    1. getData: This function gets the data from the "syncro_data.csv" or the "filter_syncro_data_valitation" file.

The data processing part is divided in 3 functions:
    1. filterData: This function limits the lidar data to a maximum value of 5 meters.
    2. polar2xy: This function converts the polar coordinates of the lidar data to cartesian coordinates.
    3. plot_lines: This function plots the lidar data in a 2D space.

The data visualization part is divided in 2 functions:
    1. plot_lines: This function plots the lidar data in a 2D space.
    2. save_image: This function saves the image in the images folder.

The script is executed by running the following command in the terminal:
> python lidar2images.py


@author: andres
@author: Felipe-Tommaselli
"""

from sys import platform

import matplotlib.pyplot as plt
import numpy as np 
import numpy as np
import os


global SLASH
if platform == "linux" or platform == "linux2":
    # linux
    SLASH = "/"
elif platform == "win32":
    # Windows...
    SLASH = "\\"

global filename 
global folder
filename = "Lidar_Data.csv"
folder = "assets" + SLASH + "tags"

''' There are other options:
folder = Dataset:
    "filter_syncro_data_validation.csv"
    "filter_syncro_data_norm.csv"
folder = Tags:
    "Label_Data.csv"
    "Lidar_Data.csv" '''
class lidar2images:
    """ Class convert the lidar data to images with each step of the lidar data (angle and distance) been converted to a point in a 2D space. """
    
    def __init__(self, filename: str, folder: str) -> None:
        """ Constructor of the class. """
        while True:
            try:
                self.data = lidar2images.getData(name=filename, folder=folder)
                break
            except Exception as e:
                print("Error: ", e)
                print("File not found, please try again")
                filename = input("Please enter the file name: ")

    @staticmethod
    def getData(name: str, folder: str) -> list:
        """ This function gets the data from the "syncro_data.csv" or the "filter_syncro_data_valitation" file. """
        # move from root (\src) to \assets\tags or \datasets
        if os.getcwd().split(SLASH)[-1] == 'src':
            os.chdir('..') 
        path = os.getcwd() + SLASH + str(folder) + SLASH
        data_file_name = os.path.join(path, name) # merge path and filename

        # open file and read all data if possible
        print("Opening file: ", data_file_name, end=' ')
        if os.path.exists(data_file_name):
            filedata = open(data_file_name,"r")
            print('[SUCESS]')
        else:
            print("[ERROR]\nFile not found, we are going to create a new one")
            filedata = open(path,"w+")

        data = filedata.readlines()
        filedata.close()
        return data # returns file data

    @staticmethod
    def filterData(readings) -> list:
        """ This function normalizes data and limits the lidar data to a maximum value of 5 meters. """
        #readings = [float(e) for e in readings[7:1088] if e != ''] # removing empty values
        readings = list(map(lambda s: s.replace('\"', '').strip(), readings)) # remove \n and others
        readings = list(map(float, readings[7:1088])) # convert to float
        
        # if mean of readings is higher than 10, the normalization is necessary
        if float(np.mean(readings)) > 10.0:
            final_readings = [float(r)/1000 for r in readings if float(r)/1000 < 5] # normalizing the data
        else: 
            final_readings = [float(r) for r in readings if float(r) < 5]
        return final_readings

    @staticmethod
    def polar2xy(lidar) -> list:
        """ This function converts the polar coordinates of the lidar data to cartesian coordinates."""
        min_angle = np.deg2rad(-45)
        max_angle = np.deg2rad(225) # lidar range
        angle = np.linspace(min_angle, max_angle, 1081, endpoint = False)

        # convert polar to cartesian:
        # x = r * cos(theta)
        # y = r * sin(theta)
        # where r is the distance from the lidar (x in lidar)
        # and angle is the step between the angles measure in each distance (angle(lidar.index(x))
        x_lidar = [x*np.cos(angle[lidar.index(x)]) for x in lidar]
        y_lidar = [y*np.sin(angle[lidar.index(y)]) for y in lidar]

        return x_lidar, y_lidar

    @staticmethod
    def plot_lines(xl: list, yl: list, t: int) -> None:
        """ This function plots the lidar data in a 2D space. """
        LW=0.8 # distance for the plot (region avaiable for navigation) 
        plt.cla() 
        plt.plot(xl,yl,'g.')
        plt.xlim(-LW,LW)
        plt.ylim(-1,5)
        plt.axis('off')
        plt.pause(0.1)
        print(f'[{t}]')
        path = ''. join([os.getcwd(), SLASH, 'assets', SLASH, 'images', SLASH])
        plt.savefig(path + str(t))

    @staticmethod
    def save_image(t: int) -> None:
        """ This function saves the image in the assets/images folder. """
        path = ''. join([os.getcwd(), SLASH, 'assets', SLASH, 'test', SLASH])
        plt.savefig(path + 'image'+str(t)+'.png')

if __name__ == '__main__':
    l2i = lidar2images(filename=filename, folder=folder)
    print('L2I OG')
    for step in range(0,len(l2i.data)):
        # split data (each line) in a lista with all the values
        readings = l2i.data[step].split(",")
        # filter data
        lidar_readings = l2i.filterData(readings=readings)
        # convert polar to cartesian
        x,y = l2i.polar2xy(lidar=lidar_readings) 
        # plot image
        l2i.plot_lines(xl=x, yl=y, t=step)
        # save image
        l2i.save_image(t=step)