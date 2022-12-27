# -*- coding: utf-8 -*-
"""

@author: andres
@author: Felipe-Tommaselli
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

import tkinter
import tkinter.ttk as ttk
from matplotlib.figure import Figure
import customtkinter as ctk

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from lidar2images import *



class lidar_tag:

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

        self.fig = Figure(figsize = (5, 5), dpi = 130)
        self.ax = self.fig.add_subplot(111)
        self.canvas = None


    def getLabel(self, raw: list, data: list) -> list:
        label_data = []
        if len(label_data) <= 1: 
            # full of empty labels
            label_data = ['' for i in range(len(data))]
        else:
            # fill the label_data with the labels if possible
            for t in range(1,len(data)):
                label_data[t] = (raw[t])[:len(raw[t])] if t < len(raw) - 1 else ''
        print('Data is ok to be tagged')  if len(label_data) == len(data) else print('Data is not ok to be tagged')
        return label_data


    #* FUNCTIONS FOR THE GUI
    def NextFunction(self) -> None:
        print('[NEXT STEP]: ' + str(self.step + 1) + ' of ' + str(self.max_step))
        self.step += 1
        if (self.step < self.max_step):
            self.PlotFunction(self.step)
        else:
            print('you reach the maximal step')


    def PreviousFunction(self) -> None:
        print('[PREVIOUS STEP]: ' + str(self.step - 1) + ' of ' + str(self.max_step))
        if (self.step <= self.min_step):
            print('you reach the minimal step')
        else:
            self.step -= 1
            self.PlotFunction(self.step)


    def GoFunction(self) -> None:
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
        print('[CLEAN]')
        self.points = []
        self.points_x = [0, 0, 0, 0]
        self.points_y = [0, 0, 0, 0]
        self.n_p = 0
        self.PlotFunction(self.step)
        self.label_data[self.step] = ''


    def SaveFunction(self):
        IL = f'step, L_x0, L_y0, L_x1, L_y1, L_x2, L_y2, L_x3, L_y3'
        if  os.getcwd().split('\\')[-1] != 'IC_NN_Lidar':
            os.chdir('..')
        path = os.getcwd() + '\\' + str(self.folder) + '\\'
        label_file_path = os.path.join(path, self.label_name) 

        os.remove(label_file_path)
        label_file = open(label_file_path,'w', encoding="utf-8")
        self.label_data[0] = IL

        for e in self.label_data:
            if e != '':
                label_file.writelines(e + '\n') 
        print('File saved: ', label_file_path)


    def PlotFunction(self, i):
        self.n_p = 0
        # split data (each line) in a lista with all the values
        lidar = ((self.lidar_data[i]).split(','))[1:]
        # filter data
        lidar_readings = lidar2images.filterData(readings=lidar)
        # convert polar to cartesian
        x_lidar,y_lidar = lidar2images.polar2xy(lidar=lidar_readings) 

        # adding the subplot
        self.ax.cla()
        # plotting the graph
        self.ax.plot(x_lidar,y_lidar,'.', color='#40b255',picker=3)
        #! TESTE COM SEABORN
        # sns.scatterplot(x=x_lidar, y=y_lidar, color='r')
        #plt.show()
        self.ax.set_title('Step: ' + str(i))
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([0, 3])

        # creating the Tkinter canvas containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.root)
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().place(x=50,y=50)

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        # creating the Matplotlib toolbar
        # toolbar = NavigationToolbar2Tk(self.canvas, root)
        # toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().place(x=40,y=40)


    def on_pick(self, event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
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
    lt = lidar_tag(lidar_name='Lidar_Data.csv', label_name='Label_Data.csv', folder='assets\\Tags')
    root, InputStep, Bnext, Bprev, Bgo, Bcln, Bsave = lt.createWindow()

    Bprev.place(x=74, y = 740)
    Bnext.place(x=177, y = 740)
    InputStep.place(x=307, y = 741)
    Bgo.place(x=367, y = 740)
    Bsave.place(x=464, y = 740)
    Bcln.place(x=567, y = 740)
    root.mainloop()