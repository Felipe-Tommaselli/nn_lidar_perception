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
import numpy as np 
import numpy as np
import os

class lidar2images:
    """ Class convert the lidar data to images with each step of the lidar data (angle and distance) been converted to a point in a 2D space. """
    
    def __init__(self) -> None:
        """ Constructor of the class. """
        while True:
            try: 
                self.data = self.getData()
                break
            except Exception as e:
                print("Error: ", e)
                print("Please, check the path to the file and try again")
                break

    def getData(self) -> list:
        """ This function gets the data from the "syncro_data.csv" or the "filter_syncro_data_valitation" file. """
        # move from root (\src) to \assets\tags
        os.chdir('..') 
        path = os.getcwd() + '\\datasets\\'
        filename = 'syncro_data_validation.csv'
        # filename = filter_syncro_data_validation.csv
        data_file_name = os.path.join(path, filename) # merge path and filename

        # open file and read all data
        filedata = open(data_file_name,"r")
        data = filedata.readlines()
        filedata.close()
        return data # returns file data

    def filterData(self, readings):
        """ This function normalizes data and limits the lidar data to a maximum value of 5 meters. """
        readings = readings[7:1088] # limit data 
        final_readings = [int(r)/1000 for r in readings if int(r)/1000 < 5] # normalizing the data
        return final_readings

    def polar2xy(self, lidar):
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

    def plot_lines(self, xl: list, yl: list, t: int):
        """ This function plots the lidar data in a 2D space. """
        LW=0.8 # distance for the plot (region avaiable for navigation) 
        plt.cla() 
        plt.plot(xl,yl,'g.')
        plt.xlim(-LW,LW)
        plt.ylim(-1,5)
        plt.axis('off')
        plt.pause(0.1)
        print(f'[{t}]')
        plt.savefig(os.getcwd() + '\\assets\\images' + '\\image' + str(t))

if __name__ == '__main__':
    l2i = lidar2images()
    for step in range(0,len(l2i.data)):
        # split data (each line) in a lista with all the values
        readings = l2i.data[step].split(",")
        # filter data
        lidar_readings = l2i.filterData(readings=readings)
        # convert polar to cartesian
        x,y = l2i.polar2xy(lidar=lidar_readings) 
        # plot image
        l2i.plot_lines(xl=x, yl=y, t=step)