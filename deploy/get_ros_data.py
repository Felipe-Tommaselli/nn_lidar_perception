#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan

def lidar_callback(data):
    # This callback function will be called whenever a new message is received on the "/terrasentia/scan" topic
    distances = data.ranges
    # Process the 'distances' list as needed
    print(distances)

def lidar_subscriber():
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber('/terrasentia/scan', LaserScan, lidar_callback)
    rospy.spin()

if __name__ == '__main__':
    lidar_subscriber()
