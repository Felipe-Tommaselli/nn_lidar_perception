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

# move from root (\src) to \assets\tags
os.chdir('..') 
path = os.getcwd() + '\\assets\\tags'


# getting file paths right
label_file_name = os.path.join(path,'Label_Data.csv')
lidar_file_name = os.path.join(path, 'Lidar_Data.csv')

#! LIDAR DATA
# open lidar file and lidar data
lidar_file = open(lidar_file_name,'r')
lidar_data = lidar_file.readlines()
Final_label_data = []

a = []
Fc = 0
#! LABEL DATA
# if label file does not exist, create it
if os.path.exists(label_file_name) == 0:
    label_file = open(label_file_name,'w', encoding="utf-8") # write = create
    print('No File')
    Fc = 1
    step = 0
    # cover all lidar data with empty labels
    for t in range(0,len(lidar_data)-1):
        # add empty label to label data
        Final_label_data.append('')
else:
    # open label file and label data
    label_file = open(label_file_name,'r', encoding="utf-8") # read
    label_data = label_file.readlines() # store label data
    label_file.close() # clos efile

    # print('label> ', label_data)
    
    # get label data length
    label_len = len(label_data) # init: label_len = 1
    step = label_len - 1
    # cover lidar_data 
    for t in range(1,len(lidar_data)):
        # if label data is not empty
        if t < label_len:
            # labelt = last element (list) in the label_data 
            labelt = (label_data[t])[:len(label_data[t])]
            # add label to label data
            Final_label_data.append(labelt)
            print('?> ', labelt)
        else:
            Final_label_data.append('')


x_lidar = []
y_lidar = []
min_angle = np.deg2rad(-45)
max_angle = np.deg2rad(225)
# angle is a list of 1081 values from -45 to 225
angle = np.linspace(min_angle, max_angle, 1081, endpoint = False)

max_step = len(lidar_data)
min_step = 1
points = []
points_x = [0, 0, 0, 0]
points_y = [0, 0, 0, 0]
n_p = 0

#! TKINTER
def NextFunction():
    print('NEXT STEP')
    global step
    global max_step
    step = step+1
    if (step<max_step):
        PlotFunction(step)
    else:
        print('you reach the maximal step')

def PreviousFunction():
    print('PREVIOUS STEP')
    global step
    global max_step
    if (step <= min_step):
        print('you reach the minimal step')
    else:
        step = step-1
        PlotFunction(step)

def GoFunction():
    INPUT = InputStep.get("1.0", "end-1c")
    global min_step
    global max_step
    global step
    if(INPUT.isnumeric() == False):
        print('It is empty or it is not a number')
    elif (int(INPUT) < min_step):
        print('The minimal step is 1')
    elif (int(INPUT) > max_step):
        print('The maximal step is '+str(max_step))
    else:
        print('it is a number')
        step = int(INPUT)
        PlotFunction(step)

def CleanFunction():
    global points_x
    global points_y
    global points
    global step, Final_label_data
    points = []
    points_x = [0, 0, 0, 0]
    points_y = [0, 0, 0, 0]
    PlotFunction(step)
    Final_label_data[step] = ''

def SaveFunction():
    global Final_label_data
    IL = 'L_x0'+','+'L_y0'+','+'L_x1'+','+'L_y1'+','+'L_x2'+','+'L_y2'+','+'L_x3'+','+'L_y3'
    #if Fc == 0:
    os.remove(label_file_name)
    print('your file was deleted')
    label_file = open(label_file_name,'w', encoding="utf-8")
    print('your files was created')
    Final_label_data[0] = IL
    for k in range (0,len(Final_label_data)):
        if Final_label_data[k] != '':
            print(Final_label_data[k]+'\n')
            label_file.writelines(Final_label_data[k]+'\n')
    print('Your doc was saved')
    


fig = Figure(figsize = (5, 5), dpi = 130)
ax = fig.add_subplot(111)

def PlotFunction(i):
    # the figure that will contain the plot
    global fig
    global ax
    global points
    global n_p
    global points_x, points_y
    points_x = [0, 0, 0, 0]
    points_y = [0, 0, 0, 0]
    points = []
    n_p = 0
    lidar = ((lidar_data[i]).split(','))[1:]
    x_lidar = []
    y_lidar = []
    global canvas
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
    
    #plt.plot(x_lidar,y_lidar)
    # list of squares
    #y = [i**2 for i in range(101)]
  
    # adding the subplot
    ax.cla()
  
    # plotting the graph
    ax.plot(x_lidar,y_lidar,'.', color='g',picker=3)
    #! TESTE COM SEABORN
    sns.scatterplot(x=x_lidar, y=y_lidar, color='r')
    plt.show()
    ax.set_title('Step: ' + str(i))
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 3])
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master = root)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().place(x=50,y=50)
    fig.canvas.mpl_connect('pick_event', on_pick)
    # creating the Matplotlib toolbar
    #toolbar = NavigationToolbar2Tk(canvas, root)
    #toolbar.update()
  
    # placing the toolbar on the Tkinter window
    #canvas.get_tk_widget().place(x=40,y=40)

root = tkinter.Tk()
root.geometry('800x800')
InputStep = tkinter.Text(root, height = 2, width = 10)
BN=tkinter.Button(root, text = 'Next', command = NextFunction)
BP=tkinter.Button(root, text = 'Previous', command = PreviousFunction)
BC=tkinter.Button(root, text = 'Go', command = GoFunction)
BCl=tkinter.Button(root, text = 'Clean', command = CleanFunction)
BS=tkinter.Button(root, text = 'Save', command = SaveFunction)
def on_pick(event):
    global points
    global n_p
    global points_x, points_y, Final_label_data
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    
    if n_p < 4:
        x1 = np.take(xdata, ind)[0]
        y1 = np.take(ydata, ind)[0]
        points.append([x1, y1])
        points_x[n_p] = x1
        points_y[n_p] = y1
        n_p = n_p +1
        print (f'X= {x1:.2f}') # Print X point
        print (f'Y={y1:.2f}')# Print Y point
        print(f'Pointsx: ' + f', '.join(f'{p:.2f}' for p in points_x))
        print(f'Pointsy: ' + f', '.join(f'{p:.2f}' for p in points_y))
        ax.plot(x1,y1,'k*')
        canvas = FigureCanvasTkAgg(fig, master = root)  
        canvas.draw()
        canvas.get_tk_widget().place(x=50,y=50)
        fig.canvas.mpl_connect('pick_event', on_pick)
        # print(points)
        
    else:
        print("IT IS NOT POSSIBLE TO SAVE MORE THAN 4 POINTS")
        left_xpoints = points_x[0:2]
        left_ypoints = points_y[0:2]
        right_xpoints = points_x[2:4]
        right_ypoints = points_y[2:4]  
        data_s = str(points_x[0])+','+str(points_y[0])+','+str(points_x[1])+','+str(points_y[1])+','+str(points_x[2])+','+str(points_y[2])+','+str(points_x[3])+','+str(points_y[3])
        Final_label_data[step] = data_s         
        ax.plot(left_xpoints,left_ypoints,'r', linewidth=2)
        ax.plot(right_xpoints,right_ypoints,'r',linewidth=2)
        canvas = FigureCanvasTkAgg(fig, master = root)  
        canvas.draw()
        canvas.get_tk_widget().place(x=50,y=50)
        fig.canvas.mpl_connect('pick_event', on_pick)            
        
    
    



#cid = fig.canvas.mpl_connect('pick_event', on_pick)




BN.place(x=200, y = 750)
BP.place(x=100, y = 750)
BC.place(x=400, y = 750)
BCl.place(x=600, y = 750)
BS.place(x=500, y = 750)
InputStep.place(x=300, y = 745)
root.mainloop()




'''
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





