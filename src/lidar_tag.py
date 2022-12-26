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
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from lidar2images import *



class lidar_tag:

    def __init__(self, lidar_name:str, label_name:str, folder:str) -> None:
        self.lidar_name = lidar_name
        self.label_name = label_name
        self.folder = folder
        self.step = 0
        self.min_step = 0
        self.max_step = 0
        self.points_x = [0, 0, 0, 0]
        self.points_y = [0, 0, 0, 0]
        self.points = []
        self.n_p = 0

        # raw_lidar_data == lidar_data
        self.lidar_data = lidar2images.getData(name=self.lidar_name, folder=self.folder)
        # raw_lidar_data != lidar_data, it requires a treatment
        raw_label_data = lidar2images.getData(name=self.label_name, folder=self.folder)
        self.label_data = self.getLabel(raw=raw_label_data, data=self.lidar_data)

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

    def NextFunction(self) -> None:
        print('[NEXT STEP]')
        self.step += 1
        if (self.step < self.max_step):
            self.PlotFunction(self.step)
        else:
            print('you reach the maximal step')

    def PreviousFunction(self) -> None:
        print('[PREVIOUS STEP]')
        if (self.step <= self.min_step):
            print('you reach the minimal step')
        else:
            self.step -= 1
            self.PlotFunction(self.step)

    def GoFunction(self) -> None:
        INPUT = InputStep.get("1.0", "end-1c")
        if(INPUT.isnumeric() == False):
            print('It is empty or it is not a number')
        elif (int(INPUT) < self.min_step):
            print('The minimal step is 1')
        elif (int(INPUT) > self.max_step):
            print('The maximal step is '+ str(self.max_step))
        else:
            print('it is a number')
            step = int(INPUT)
            self.PlotFunction(self.step)

    def CleanFunction(self) -> None:
        self.points = []
        self.points_x = [0, 0, 0, 0]
        self.points_y = [0, 0, 0, 0]
        self.label_data[self.step] = ''
        self.PlotFunction(self.step)

    def SaveFunction(self):
        IL = 'L_x0'+','+'L_y0'+','+'L_x1'+','+'L_y1'+','+'L_x2'+','+'L_y2'+','+'L_x3'+','+'L_y3'
        if  os.getcwd().split('\\')[-1] != 'IC_NN_Lidar':
            os.chdir('..')
        path = os.getcwd() + '\\' + str(self.folder) + '\\'
        label_file_name = os.path.join(path, self.label_name) 

        os.remove(label_file_name)
        print('old file was deleted')
        label_file = open(label_file_name,'w', encoding="utf-8")
        print('new files was created')
        self.label_data[0] = IL
        for e in self.label_data:
            if e != '':
                label_file.writelines(e + '\n') 
        print('File saved :)')

    def PlotFunction(self, i):
        # reset values
        self.points_x = [0, 0, 0, 0]
        self.points_y = [0, 0, 0, 0]
        self.points = []
        self.n_p = 0
        lidar = ((self.lidar_data[i]).split(','))[1:]
        x_lidar = []
        y_lidar = []
        
        for j in range(0,len(lidar)):
            if j==0:
                x_lidar.append(float((lidar[j])[1:])*math.cos(angle[j]))
                y_lidar.append(float((lidar[j])[1:])*math.sin(angle[j]))
            elif j==len(lidar)-1:
                x_lidar.append(float((lidar[j])[:len(lidar[j])-2])*math.cos(angle[j]))
                y_lidar.append(float((lidar[j])[:len(lidar[j])-2])*math.sin(angle[j]))
            else:
                x_lidar.append(float(lidar[j])*math.cos(angle[j]))
                y_lidar.append(float(lidar[j])*math.sin(angle[j]))
        
        #plt.plot(x_lidar,y_lidar)
        # list of squares
        #y = [i**2 for i in range(101)]
    
        # adding the subplot
        self.ax.cla()
    
        # plotting the graph
        self.ax.plot(x_lidar,y_lidar,'.', color='g',picker=3)
        #! TESTE COM SEABORN
        sns.scatterplot(x=x_lidar, y=y_lidar, color='r')
        plt.show()
        self.ax.set_title('Step: ' + str(i))
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([0, 3])
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(fig, master = root)  
        self.canvas.draw()
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().place(x=50,y=50)
        self.fig.canvas.mpl_connect('pick_event', on_pick)
        # creating the Matplotlib toolbar
        #toolbar = NavigationToolbar2Tk(canvas, root)
        #toolbar.update()
    
        # placing the toolbar on the Tkinter window
        #canvas.get_tk_widget().place(x=40,y=40)

    def createWindow(self):
        self.root = tkinter.Tk()
        self.root.geometry('800x800')
        InputStep = tkinter.Text(self.root, height = 2, width = 10)
        BN  = tkinter.Button(self.root, text = 'Next', command = self.NextFunction)
        BP  = tkinter.Button(self.root, text = 'Previous', command = self.PreviousFunction)
        BC  = tkinter.Button(self.root, text = 'Go', command = self.GoFunction)
        BCl = tkinter.Button(self.root, text = 'Clean', command = self.CleanFunction)
        BS  = tkinter.Button(self.root, text = 'Save', command = self.SaveFunction)
        return self.root, InputStep, BN, BP, BC, BCl, BS

    def on_pick(event):
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
            print (f'X= {x1:.2f}') # Print X point
            print (f'Y={y1:.2f}')# Print Y point
            print(f'Pointsx: ' + f', '.join(f'{p:.2f}' for p in self.points_x))
            print(f'Pointsy: ' + f', '.join(f'{p:.2f}' for p in self.points_y))
            self.ax.plot(x1,y1,'k*')
            self.canvas = FigureCanvasTkAgg(fig, master = root)  
            self.canvas.draw()
            self.canvas.get_tk_widget().place(x=50,y=50)
            self.fig.canvas.mpl_connect('pick_event', on_pick)
            # print(points)
            
        else:
            print("IT IS NOT POSSIBLE TO SAVE MORE THAN 4 POINTS")
            left_xpoints = self.points_x[0:2]
            left_ypoints = self.points_y[0:2]
            right_xpoints = self.points_x[2:4]
            right_ypoints = self.points_y[2:4]  
            data_s = f'{str(self.points_x[0])}, {str(self.points_y[0])},  {str(self.points_x[1])}, {str(self.points_y[1])}, {str(self.points_x[2])}, {str(self.points_y[2])}, {str(self.points_x[3])}, {str(self.points_y[3])}'
            self.label_data[step] = data_s         
            self.ax.plot(left_xpoints,left_ypoints,'r', linewidth=2)
            self.ax.plot(right_xpoints,right_ypoints,'r',linewidth=2)
            self.canvas = FigureCanvasTkAgg(fig, master = root)  
            self.canvas.draw()
            self.canvas.get_tk_widget().place(x=50,y=50)
            self.fig.canvas.mpl_connect('pick_event', on_pick)            


if __name__ == '__main__':
    lt = lidar_tag(lidar_name='Lidar_Data.csv', label_name='Label_Data.csv', folder='assets\\Tags')
    root, InputStep, BN, BP, BC, BCl, BS = lt.createWindow()

    BN.place(x=200, y = 750)
    BP.place(x=100, y = 750)
    BC.place(x=400, y = 750)
    BCl.place(x=600, y = 750)
    BS.place(x=500, y = 750)
    InputStep.place(x=300, y = 745)
    root.mainloop()


'''
#! TKINTER

#cid = fig.canvas.mpl_connect('pick_event', on_pick)

for i in range(1, len(lidar_data)):
    lidar = ((lidar_data[i]).split(','))[1:]
    x_lidar = []
    y_lidar = []
    print(len(lidar))
    for j in range(0,len(lidar)):
        #print(j)
        if (j==0):
            x_lidar.append(float((lidar[j])[1:])*math.cos(angle[j]))
            y_lidar.append(float((lidar[j])[1:])*math.sin(angle[j]))
        elif (j==len(lidar)-1):
            x_lidar.append(float((lidar[j])[:len(lidar[j])-2])*math.cos(angle[j]))
            y_lidar.append(float((lidar[j])[:len(lidar[j])-2])*math.sin(angle[j]))
        else:
            x_lidar.append(float(lidar[j])*math.cos(angle[j]))
            y_lidar.append(float(lidar[j])*math.sin(angle[j]))
    
    plt.plot(x_lidar,y_lidar)
    plt.pause(0.001)

'''





