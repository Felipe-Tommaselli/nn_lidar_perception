# -*- coding: utf-8 -*-
"""
lidar_tag.py is a script that make avaiable the UI to tags the lidar data for the Neural Network. This script is responsable for converting the 
data to images and make manually labeling with de UI developed possible.

The script was planned around the UI platform for image labeling, the UI was developed with the tkinter library with the wrapper customtkinter. 
The UI is divided in 3 parts:
    1. The canvas where the images are plotted.
    2. The buttons to navigate through the images.
    3. The entry to point the label on the image.

The script is executed by running the following command in the terminal:
> python lidar_tag.py

@author: andres
@author: Felipe-Tommaselli
"""

import os
from sys import platform
import shutil

import numpy as np
import math
import matplotlib.pyplot as plt

import tkinter
import tkinter.ttk as ttk
from matplotlib.figure import Figure
import customtkinter as ctk

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from lidar2images import *

global POINT_WIDTH
POINT_WIDTH = 10

global SLASH
if platform == "linux" or platform == "linux2":
    # linux
    SLASH = "/"
elif platform == "win32":
    # Windows...
    SLASH = "\\"

class lidar_tag:
    """ Class that make avaiable the UI to tags the lidar data for the Neural Network. """
    def __init__(self, lidar_name:str, label_name:str, folder:str) -> None:
        self.lidar_name = lidar_name
        self.label_name = label_name
        self.folder = folder
        # raw_lidar_data == lidar_data
        self.lidar_data = lidar2images.getData(name=self.lidar_name, folder=self.folder)
        # raw_lidar_data != lidar_data, it requires a treatment
        raw_label_data = lidar2images.getData(name=self.label_name, folder=self.folder)
        self.label_data = self.getLabel(raw=raw_label_data, data=self.lidar_data)

        self.step = 0
        self.min_step = 0
        self.max_step = len(self.lidar_data)
        self.points_x = [0, 0, 0, 0]
        self.points_y = [0, 0, 0, 0]
        self.points = []
        self.n_p = 0

        self.fig_holding = Figure(figsize = (5, 5), dpi = 130)
        self.fig = Figure(figsize = (5, 5), dpi = 130)
        self.ax = self.fig.add_subplot(111)
        self.canvas = None


    def getLabel(self, raw: list, data: list) -> list:
        """ Function that get the labels from the raw data and fill the label_data with the labels if possible. """
        IL = f'step, L_x0, L_y0, L_x1, L_y1, L_x2, L_y2, L_x3, L_y3'
        label_data = [r if r.index != 0 else IL for r in raw]

        if len(raw) <= 1: 
            # full of empty labels
            label_data = ['' if r.index != 0 else IL for r in data]
        else:
            # concatena label data (from raw) with empty labels for the data that is not labeled yet
            label_data += ['' for i in range(len(raw), len(data))]
        return label_data


    #* FUNCTIONS FOR THE GUI
    def NextFunction(self) -> None:
        """ Function that change to next image for classification on click of the button "Next". """
        print('[NEXT STEP]: ' + str(self.step + 1) + ' of ' + str(self.max_step))
        self.step += 1
        if (self.step < self.max_step):
            self.PlotFunction(self.step)
        else:
            print('you reach the maximal step')


    def PreviousFunction(self) -> None:
        """ Function that change to previous image for classification on click of the button "Previous". """
        print('[PREVIOUS STEP]: ' + str(self.step - 1) + ' of ' + str(self.max_step))
        if (self.step <= self.min_step):
            print('you reach the minimal step')
        else:
            self.step -= 1
            self.PlotFunction(self.step)


    def GoFunction(self) -> None:
        """ Function that change to the desired image for classification on click of the button "Go". """
        INPUT = InputStep.get("1.0", "end-1c")
        if(INPUT.isnumeric() == False):
            print('[ERROR] It is empty or it is not a number')
        elif (int(INPUT) < self.min_step):
            print('[ERROR] The minimal step is 1')
        elif (int(INPUT) > self.max_step):
            print('[ERROR] The maximal step is '+ str(self.max_step))
        else:
            print(f'[GO TO STEP]: {INPUT} of {self.max_step}')
            self.step = int(INPUT)
            self.PlotFunction(self.step)


    def CleanFunction(self) -> None:
        """ Function that clean the points on click of the button "Clean". """
        print('[CLEAN]')
        self.points = []
        self.points_x = [0, 0, 0, 0]
        self.points_y = [0, 0, 0, 0]
        self.n_p = 0
        self.PlotFunction(self.step)
        self.label_data[self.step] = ''


    def SaveFunction(self):
        """ Function that save the points on click of the button "Save". """
        if  os.getcwd().split(SLASH)[-1] != 'IC_NN_Lidar':
            os.chdir('..')
        path = os.getcwd() + SLASH + str(self.folder) + SLASH
        label_file_path = os.path.join(path, self.label_name) 

        label_file = open(label_file_path, 'r')
        text = label_file.readlines()
        label_file.close()

        label_file = open(label_file_path, 'a')
        label_file.write('\n' + self.label_data[self.step])
        label_file.close()
        print('File saved: ', label_file_path)

        # copy the image on the step to the folder of the images that are already classified
        if os.getcwd().split(SLASH)[-1] == 'src':
            os.chdir('..') 
        folder_class = os.getcwd() + SLASH + 'assets' + SLASH + 'classified' + SLASH

        if not os.path.exists(folder):
            os.makedirs(os.getcwd() + 'classified')
        # sabe matplotlib plot on folder_class
        self.fig_holding.savefig(folder_class + 'image' + str(self.step) + '.png')

    def PlotFunction(self, i):
        """ Function that plot the image that is going to be classified."""
        self.n_p = 0
        # split data (each line) in a lista with all the values
        lidar = ((self.lidar_data[i]).split(','))[1:]
        # filter data
        lidar_readings = lidar2images.filterData(readings=lidar)
        # convert polar to cartesian
        self.x_lidar, self.y_lidar = lidar2images.polar2xy(lidar=lidar_readings) 

        # adding the subplot
        self.ax.cla()
        # plotting the graph
        self.ax.plot(self.x_lidar,self.y_lidar, '.', markersize=POINT_WIDTH, color='#40b255',picker=3)

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title('Step: ' + str(i))
        self.ax.set_xlim([-1.0, 1.2])
        self.ax.set_ylim([-0.25, 3])
        self.ax.grid(False)
        
        self.fig_holding = self.fig

        # creating the Tkinter canvas containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.root)
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().place(x=50,y=50)

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        # creating the Matplotlib toolbar
        # toolbar = NavigationToolbar2Tk(self.canvas, root)
        # toolbar.update()    


    def on_pick(self, event):
        """ Function that get the points that will make the labeling """
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        # xdata = self.x_lidar[ind]
        # ydata = self.y_lidar[ind]
        ind = event.ind

        if self.n_p < 4:
            x1 = np.take(xdata, ind)[0]
            y1 = np.take(ydata, ind)[0]
            self.points.append([x1, y1])
            self.points_x[self.n_p] = x1
            self.points_y[self.n_p] = y1
            self.n_p += 1
            # print (f'X= {x1:.2f}') # Print X point
            # print (f'Y={y1:.2f}')# Print Y point
            # print(f'Pointsx: ' + f', '.join(f'{p:.2f}' for p in self.points_x))
            # print(f'Pointsy: ' + f', '.join(f'{p:.2f}' for p in self.points_y))
            self.ax.plot(x1,y1,'k*')
            self.canvas = FigureCanvasTkAgg(self.fig, master = root)  
            self.canvas.draw()
            self.canvas.get_tk_widget().place(x=50,y=50)
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        else:
            print('Lines are already drawn!') if self.n_p == 4 else print("IT IS NOT POSSIBLE TO SAVE MORE THAN 4 POINTS")
            left_xpoints = self.points_x[0:2]
            left_ypoints = self.points_y[0:2]
            right_xpoints = self.points_x[2:4]
            right_ypoints = self.points_y[2:4]  
            data_s = f'{str(self.step)}, {str(self.points_x[0])}, {str(self.points_y[0])},  {str(self.points_x[1])}, {str(self.points_y[1])}, {str(self.points_x[2])}, {str(self.points_y[2])}, {str(self.points_x[3])}, {str(self.points_y[3])}'
            self.label_data[self.step] = data_s         
            self.ax.plot(left_xpoints,left_ypoints,'r', linewidth=2)
            self.ax.plot(right_xpoints,right_ypoints,'r',linewidth=2)
            self.canvas = FigureCanvasTkAgg(self.fig, master = root)  
            self.canvas.draw()
            self.canvas.get_tk_widget().place(x=50,y=50)
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)            


    def createWindow(self):
        """ Function that create the window of the application. """
        self.root = ctk.CTk()
        self.root.geometry('740x800')
        self.root.title('Lidar Labeling Tool')
        ctk.set_appearance_mode("dark")
        
        # configure and customise all custom tkinter buttons with configure

        Bprev  = ctk.CTkButton(self.root, text = 'Previous', command = self.PreviousFunction,
                            width = 100, height=35, fg_color='#349b47', font=('Arial', 14, 'bold'))
        Bnext  = ctk.CTkButton(self.root, text = 'Next', command = self.NextFunction, 
                            width = 100, height=35, fg_color='#349b47', font=('Arial', 14, 'bold'))

        InputStep = ctk.CTkTextbox(self.root, height = 4, width = 57, font=('Arial', 14))
        Bgo  = ctk.CTkButton(self.root, text = 'Go', command = self.GoFunction,
                            width = 70, height=35, fg_color='#349b47', font=('Arial', 14, 'bold'))
        Bcln = ctk.CTkButton(self.root, text = 'Clean', command = self.CleanFunction,
                            width = 100, height=35, fg_color='#349b47', font=('Arial', 14, 'bold'))
        Bsave  = ctk.CTkButton(self.root, text = 'Save', command = self.SaveFunction,
                            width = 100, height=35, fg_color='#349b47', font=('Arial', 14, 'bold'))
        return self.root, InputStep, Bnext, Bprev, Bgo, Bcln, Bsave


if __name__ == '__main__':
    lt = lidar_tag(lidar_name='Lidar_Data.csv', label_name='Label_Data.csv', folder=''.join(['assets', SLASH, 'tags']))
    root, InputStep, Bnext, Bprev, Bgo, Bcln, Bsave = lt.createWindow()

    Bprev.place(x=74, y = 740)
    Bnext.place(x=177, y = 740)
    InputStep.place(x=307, y = 741)
    Bgo.place(x=367, y = 740)
    Bsave.place(x=464, y = 740)
    Bcln.place(x=567, y = 740)
    root.mainloop()