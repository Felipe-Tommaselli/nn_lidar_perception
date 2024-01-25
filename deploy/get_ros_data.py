#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

import time


def lidar_callback(data, fig, line1):
    # This callback function will be called whenever a new message is received on the "/terrasentia/scan" topic
    distances = data.ranges
    y1p = np.ones(len(x))*int(np.ceil(distances[100]*100))
    print(y1p[0])

    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(y1p)

    # change title
    fig.suptitle('Updated Title')

    # drawing updated values
    fig.canvas.draw()

    fig.canvas.flush_events()
    time.sleep(0.05)

def lidar_subscriber(fig, line1):
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber('/terrasentia/scan', LaserScan, lambda data: lidar_callback(data, fig, line1))
    rospy.spin()

if __name__ == '__main__':

    ########## PLOT ########## 
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 5), frameon=True)
    x = np.arange(0, 224)
    linewidth = 2.5

    # create the lines with rand values
    line1, = ax.plot(x, x, color='red', label='Predicted', linewidth=linewidth)
    image = np.zeros((224, 224)) # empty blank (224, 224) image

    border_style = dict(facecolor='none', edgecolor='black', linewidth=2)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, **border_style, transform=ax.transAxes))
    plt.legend(loc='upper right', prop={'size': 9, 'family': 'Ubuntu'})
    ax.axis('off')
    ax.imshow(image, cmap='magma')
    lidar_subscriber(fig, line1)

