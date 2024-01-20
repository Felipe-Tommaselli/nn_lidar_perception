import rosbag
import csv
from sensor_msgs.msg import LaserScan, Image
import os

global fid 
fid = 3

os.chdir('..') 

def extract_topic_messages(bag_filename):
    bag = rosbag.Bag(bag_filename, "r")
    messages_laser = []
    for topic, msg, t in bag.read_messages():
        if topic == "/terrasentia/scan":
            messages_laser.append((t, list(msg.ranges)))
    bag.close()
    return messages_laser

def laserscan_messages_to_csv(messages, output_filename):
    with open(output_filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp"] + ["lidar (" + str(len(messages[0][1])) + ")" ])
        # write empty line
        writer.writerow([])
        for t, ranges in messages:
            row = [t.to_sec()] + ranges
            writer.writerow(row)
            writer.writerow([])

if __name__ == "__main__":
    bag_filename = "../datasets/gazebo/crop" + str(fid) + ".bag"
    
    output_filename = "../datasets/Crop_Data" + str(fid) + ".csv"
    messages_laser = extract_topic_messages(bag_filename)

    laserscan_messages_to_csv(messages_laser, output_filename)
