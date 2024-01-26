#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class LidarSubscriber:
    def __init__(self):

        ########## PLOT ##########
        self.fig, _ = plt.subplots(figsize=(8, 5), frameon=True)
        self.x = np.arange(0, 224)
        linewidth = 2.5

        # create the lines with rand values
        self.line, = plt.plot(self.x, self.x, color='red', label='Predicted', linewidth=linewidth)
        image = np.zeros((224, 224))  # empty blank (224, 224) image

        border_style = dict(facecolor='none', edgecolor='black', linewidth=2)
        plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, **border_style, transform=plt.gca().transAxes))
        plt.axis('off')
        plt.imshow(image, cmap='magma')

        ############### RUN ###############
        self.y1p = np.zeros(len(self.x))

        # Set up the ROS subscriber
        rospy.init_node('lidar_subscriber', anonymous=True)
        rospy.Subscriber('/terrasentia/scan', LaserScan, self.lidar_callback)

        # Set up the animation
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=100)  # Adjust the interval as needed

        # Show the plot
        plt.show(block=True)

    def update_plot(self, frame):
        # Update data values
        self.line.set_ydata(self.y1p)

        # Change title
        self.fig.suptitle('Updated Title')

    def lidar_callback(self, data):
        # This callback function will be called whenever a new message is received on the "/terrasentia/scan" topic
        distances = data.ranges
        try:
            if isinstance(distances[10], float):
                self.y1p = np.ones(len(self.x)) * int(np.ceil(distances[10] * 100))
        except:
            pass

if __name__ == '__main__':
    ls = LidarSubscriber()
