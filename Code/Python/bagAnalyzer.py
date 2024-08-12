import rosbag
import os
import numpy as np
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
from cor_tud_msgs.msg import CartesianState
from spatialmath import SE3
import time
import tf
import matplotlib.pyplot as plt

class bagAnalyzer():
    def __init__(self, bag_file = None):
        """
        Initialize the bag analyzer, with one or more bags
        """
        # check if the the bag_file is a list of different bags
        if type(bag_file) == list or bag_file is None:
            self.bag_file_list = bag_file
        else:
            self.bag_file_list = [bag_file]

    def check_bag(self, bag_id):
        # Open the rosbag file
        bag = rosbag.Bag(self.bag_file_list[bag_id])

        # Get the list of topics in the rosbag
        topics = bag.get_type_and_topic_info().topics.keys()

        # Iterate through each topic and print some information
        for topic in topics:
            print(f"Topic: {topic}")
            
            # Get the messages in the rosbag for the current topic
            messages = bag.read_messages(topics=[topic])
            
            # Print some information about the first message
            for msg in messages:
                print(f"  Message Type: {msg[0]}")
                print(f"  Timestamp: {msg[1]}")
                print(f"  Message Data: {msg[2]}")
                break  # Only print information about the first message

        # Close the rosbag file
        bag.close()


    def list_topics_bag(self, bag_id):
        # Open the rosbag file
        bag = rosbag.Bag(self.bag_file_list[bag_id])

        # Get the list of topics in the rosbag
        topics = bag.get_type_and_topic_info().topics.keys()

        # Iterate through each topic and print some information
        for topic in topics:
            print(f"Topic: {topic}")

            # Get the messages in the rosbag for the current topic
            messages = bag.read_messages(topics=[topic])
            
            # Print some information about the first message
            for msg in messages:
                print(f"  Message Type: {msg[0]}")
                break

        # Close the rosbag file
        bag.close()