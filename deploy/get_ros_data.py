#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

import time

class lidar_subscriber:
    def __init__(self):
        self.fig = None
        self.line1 = None
        self.ax = None

        ########## PLOT ########## 
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(8, 5), frameon=True)
        self.x = np.arange(0, 224)
        linewidth = 2.5

        # create the lines with rand values
        self.line1, = self.ax.plot(self.x, self.x, color='red', label='Predicted', linewidth=linewidth)
        image = np.zeros((224, 224)) # empty blank (224, 224) image

        border_style = dict(facecolor='none', edgecolor='black', linewidth=2)
        self.ax.add_patch(plt.Rectangle((0, 0), 1, 1, **border_style, transform=self.ax.transAxes))
        self.ax.axis('off')
        self.ax.imshow(image, cmap='magma')

        self.lidar_main()

    def lidar_callback(self, data):
        # This callback function will be called whenever a new message is received on the "/terrasentia/scan" topic
        distances = data.ranges
        y1p = np.ones(len(self.x))*int(np.ceil(distances[100]*100))
        print(y1p[0])

        # updating data values
        self.line1.set_xdata(self.x)
        self.line1.set_ydata(y1p)

        # change title
        self.fig.suptitle('Updated Title')

        # drawing updated values
        self.fig.canvas.draw()

        self.fig.canvas.flush_events()
        rospy.Rate(2).sleep()

    def lidar_main(self):
        rospy.init_node('lidar_subscriber', anonymous=True)
        rospy.Subscriber('/terrasentia/scan', LaserScan, self.lidar_callback)
        rospy.spin()

if __name__ == '__main__':
    ls = lidar_subscriber()