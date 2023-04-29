import rosbag
import csv
from sensor_msgs.msg import LaserScan, Image

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
        for t, ranges in messages:
            row = [t.to_sec()] + ranges
            writer.writerow(row)

if __name__ == "__main__":
    fid = 2
    bag_filename = "../ROSBags/ts_lidar_camera.bag"
    
    output_filename = "../datasets/Lidar_Data" + fid + ".csv"
    messages_laser = extract_topic_messages(bag_filename)

    laserscan_messages_to_csv(messages_laser, output_filename)
