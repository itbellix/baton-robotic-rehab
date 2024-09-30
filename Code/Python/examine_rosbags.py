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

def count_messages(bag_path, topic):
    count = 0
    with rosbag.Bag(bag_path, 'r') as bag:
        for _, _, _ in bag.read_messages(topics=[topic]):
            count += 1
    return count


def calculate_frequency(bag_path, topic):
    with rosbag.Bag(bag_path, 'r') as bag:
        start_time = None
        end_time = None
        message_count = 0

        for _, _, t in bag.read_messages(topics=[topic]):
            if start_time is None:
                start_time = t
            end_time = t
            message_count += 1

        if start_time is not None and end_time is not None:
            time_span = (end_time - start_time).to_sec()
            frequency = message_count / time_span
            return frequency
        else:
            return 0.0


def marker_pose_callback(marker_publisher, msg, marker_id):
    # Assuming msg.data is a 6x1 numpy array [x, y, z, roll, pitch, yaw]
    x, y, z = msg

    # Create a Marker message
    marker = Marker()
    marker.header.frame_id = "base_link"  # Set your robot's frame_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    # You may need to convert roll-pitch-yaw to quaternion and set it to marker.pose.orientation
    marker.scale.x = marker.scale.y = marker.scale.z = 0.005  # Set the size of the marker
    marker.color.a = 1.0
    marker.color.r = 1.0 if marker_id == 1 else 0.0  # Red for topic1, green for topic2
    marker.color.g = 1.0 if marker_id == 2 else 0.0
    marker.color.b = 0.0
    # marker.lifetime = rospy.Duration.from_sec(0.1)  # Adjust the lifetime

    # Publish the Marker message
    marker_publisher.publish(marker)


def marker_orient_callback(marker_publisher, msg, marker_id):
    # Assuming msg.data is a 6x1 numpy array [x, y, z, roll, pitch, yaw]
    x, y, z = msg[0:3, 3]

    orientation_matrix = msg[0:3, 0:3]

    # Convert the orientation matrix to quaternion
    orientation_quaternion = tf.transformations.quaternion_from_matrix(msg)

    # Create a Marker message
    marker = Marker()
    marker.header.frame_id = "base_link"  # Set your robot's frame_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.x = orientation_quaternion[0]
    marker.pose.orientation.y = orientation_quaternion[1]
    marker.pose.orientation.z = orientation_quaternion[2]
    marker.pose.orientation.w = orientation_quaternion[3]

    marker.scale.x = 0.02  # Set the size of the marker
    marker.scale.y =  0.004  # Set the size of the marker
    marker.scale.z = 0.004 # Set the size of the marker
    marker.color.a = 1.0
    marker.color.r = 1.0 if marker_id == 1 else 0.0  # Red for topic1, green for topic2
    marker.color.g = 1.0 if marker_id == 2 else 0.0
    marker.color.b = 0.0
    # marker.lifetime = rospy.Duration.from_sec(0.1)  # Adjust the lifetime

    # Publish the Marker message
    marker_publisher.publish(marker)

def main():

    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags')
    bag_file_name = '2024-02-14-11-44-15.bag'

    rospy.init_node('marker_publisher_node', anonymous=True)

    rate = rospy.Rate(10)

    marker1pose_publisher = rospy.Publisher('ee_cartesian_pose', Marker, queue_size=1)
    marker1orient_publisher = rospy.Publisher('ee_cartesian_orientation', Marker, queue_size=1)
    marker2pose_publisher = rospy.Publisher('desired_pose', Marker, queue_size=1)
    marker2orient_publisher = rospy.Publisher('desired_orientation', Marker, queue_size=1)

    bag_path = os.path.join(path_to_bag, bag_file_name)

    # f1 = calculate_frequency(bag_path, '/iiwa7/ee_cartesian_pose')
    # f2 = calculate_frequency(bag_path, '/optimal_cartesian_ref_ee')

    xyz_cmd = None
    xyz_curr = None
    velocity = None

    messages_topic1 = {'message':[], 'time':[]}
    messages_topic2 = {'message':[], 'time':[]}
    with rosbag.Bag(bag_path, 'r') as bag:

        print('Extracting messages from third topic')
        for _, msg, time_msg in bag.read_messages(topics=['/estimated_shoulder_pose']):
            messages_topic2['message'].append(msg)
            messages_topic2['time'].append(time_msg)
            if velocity is None:
                velocity = msg.data
            velocity = np.vstack((velocity, msg.data))
        
        # visualize velocity
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(velocity[:, 1], label='pe_dot')
        ax.plot(velocity[:, 3], label='se_dot')
        ax.plot(velocity[:, 5], label='ar_dot')
        ax.set_xlabel("Time [@ 10Hz]")
        ax.set_ylabel("velocities")
        ax.legend()

        print('Extracting messages from third topic')
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/ee_cartesian_pose']):
            messages_topic2['message'].append(msg)
            messages_topic2['time'].append(time_msg)
            if velocity is None:
                velocity = msg.velocity
            velocity = np.vstack((velocity, msg.velocity))
        
        # visualize velocity
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(velocity[:, 0], label='v_x')
        ax.plot(velocity[:, 1], label='v_y')
        ax.plot(velocity[:, 2], label='v_z')
        ax.plot(velocity[:, 3], label='omega_x')
        ax.plot(velocity[:, 4], label='omega_y')
        ax.plot(velocity[:, 5], label='omega_z')
        ax.set_xlabel("Time [@ 10Hz]")
        ax.set_ylabel("velocities")
        ax.legend()


        print('Extracting messages from first topic')
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/ee_cartesian_pose']):
            messages_topic1['message'].append(msg)
            messages_topic1['time'].append(time_msg)
            if xyz_curr is None:
                xyz_curr = np.reshape(msg.pose, (4,4))[0:3, 3]
            xyz_curr = np.vstack((xyz_curr, np.reshape(msg.pose, (4,4))[0:3, 3]))

        print('Extracting messages from second topic')
        for _, msg, time_msg in bag.read_messages(topics=['/optimal_cartesian_ref_ee']):
            messages_topic2['message'].append(msg)
            messages_topic2['time'].append(time_msg)
            if xyz_cmd is None:
                xyz_cmd = np.reshape(msg.data, (4,4))[0:3, 3]
            xyz_cmd = np.vstack((xyz_cmd, np.reshape(msg.data, (4,4))[0:3, 3]))
    
    # downsample current positions
    downsampled_indices = np.linspace(0, len(xyz_curr) - 1, len(xyz_cmd)).astype(int)
    xyz_curr = xyz_curr[downsampled_indices,:]

    # visualize msg pose
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(xyz_cmd[:, 0])
    ax.plot(xyz_curr[:, 0])
    ax.scatter(np.arange(xyz_cmd[:, 0].shape[0]), xyz_cmd[:, 0], label = 'X_cmd')    # plot commanded x
    ax.scatter(np.arange(xyz_curr[:, 0].shape[0]), xyz_curr[:, 0], label = 'X_cur')  # plot actual x
    ax.set_xlabel("Time [@ 10Hz]")
    ax.set_ylabel("m")
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(xyz_cmd[:, 1])
    ax.plot(xyz_curr[:, 1])
    ax.scatter(np.arange(xyz_cmd[:, 1].shape[0]), xyz_cmd[:, 1], label = 'Y_cmd')    
    ax.scatter(np.arange(xyz_curr[:, 1].shape[0]), xyz_curr[:, 1], label = 'Y_cur')
    ax.set_xlabel("Time [@ 10Hz]")
    ax.set_ylabel("m")
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(xyz_cmd[:, 2])
    ax.plot(xyz_curr[:, 2])
    ax.scatter(np.arange(xyz_cmd[:, 2].shape[0]), xyz_cmd[:, 2], label = 'Z_cmd')    
    ax.scatter(np.arange(xyz_curr[:, 2].shape[0]), xyz_curr[:, 2], label = 'Z_cur')
    ax.set_xlabel("Time [@ 10Hz]")
    ax.set_ylabel("m")
    ax.legend()
    
    plt.show()

    # Merge the two lists into a single list
    merged_messages = messages_topic1['message'] + messages_topic2['message']
    merged_times = messages_topic1['time'] + messages_topic2['time']

    # Sort the merged list based on timestamps
    merged_data = sorted(zip(merged_times, merged_messages), key=lambda x: x[0])

    print('Done, ready for visualization')
    print(3)
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')
    time.sleep(1)

    # Iterate through the sorted list and publish markers
    for _ , message in merged_data:
        if message in messages_topic1['message']:
            marker_pose_callback(marker1pose_publisher, np.reshape(message.pose, (4,4))[0:3, 3], marker_id=1)
            marker_orient_callback(marker1orient_publisher, np.reshape(message.pose, (4,4)), marker_id=1)
        elif message in messages_topic2['message']:
            marker_pose_callback(marker2pose_publisher, np.reshape(message.data, (4,4))[0:3, 3], marker_id=2)
            marker_orient_callback(marker2orient_publisher, np.reshape(message.data, (4,4)), marker_id=2)



if __name__ == '__main__':
    main()