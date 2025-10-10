#!/usr/bin/env python

# This script subscribes to a ROS image topic and a waypoint topic,
# then saves a matplotlib figure of the latest data every second.

import rospy
import cv2
import numpy as np
import threading
import os

# ROS Message Imports
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# Matplotlib for plotting
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# CV-Bridge to convert ROS images to OpenCV images
from cv_bridge import CvBridge, CvBridgeError

# --- Global Variables & Configuration ---

# --- IMPORTANT: Change these topic names to match your robot's topics ---
CAMERA_TOPIC = '/realsense/color/image_raw'  # Example topic
WAYPOINT_TOPIC = '/waypoint'              # Example topic
OUTPUT_DIR = '/share1/theyanesh.er/flownav_deploy'              # Directory to save figures

# Global variables to store the latest data from callbacks
latest_image = None
waypoint_history = []
bridge = CvBridge()

# Threading locks to prevent race conditions when accessing global variables
image_lock = threading.Lock()
waypoint_lock = threading.Lock()

# --- ROS Callback Functions ---

def camera_callback(msg):
    """
    Callback function for the image subscriber.
    Converts the ROS Image message to an OpenCV image and stores it.
    """
    global latest_image
    try:
        # Convert ROS Image message to OpenCV image (BGR format)
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    # Safely update the global image variable
    with image_lock:
        latest_image = cv_image

def waypoint_callback(msg):
    """
    Callback function for the waypoint subscriber.
    Stores the waypoint data for plotting.
    """
    global waypoint_history
    # Assuming the waypoint is a 2D point [x, y] from the Float32MultiArray
    if len(msg.data) >= 2:
        waypoint = (msg.data[0], msg.data[1])
        
        # Safely update the global waypoint history
        with waypoint_lock:
            waypoint_history.append(waypoint)
            # Keep the history to a manageable size (e.g., last 50 points)
            if len(waypoint_history) > 50:
                waypoint_history.pop(0)

def update_and_save_figure(fig, ax_img, ax_wp, frame_count):
    """
    Updates the plot with the latest data and saves it to a file.
    """
    global latest_image, waypoint_history

    # --- Update Camera Image ---
    current_image = None
    with image_lock:
        if latest_image is not None:
            # Matplotlib expects RGB, so convert from OpenCV's BGR
            current_image = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)
    
    # If there's an image, display it. Otherwise, show a black placeholder.
    if current_image is not None:
        ax_img.imshow(current_image)
    else:
        ax_img.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    
    ax_img.set_title('Camera Feed')
    ax_img.axis('off')

    # --- Update Waypoint Plot ---
    # Clear the previous waypoint plot to redraw it
    ax_wp.clear()
    ax_wp.set_title('Received Waypoints')
    ax_wp.set_xlabel('X Coordinate (m)')
    ax_wp.set_ylabel('Y Coordinate (m)')
    ax_wp.set_xlim(-2, 2)
    ax_wp.set_ylim(-2, 2)
    ax_wp.grid(True)
    ax_wp.set_aspect('equal', adjustable='box')

    with waypoint_lock:
        if waypoint_history:
            x_data, y_data = zip(*waypoint_history)
            ax_wp.plot(x_data, y_data, 'b.-', label='Waypoint History')
            ax_wp.plot(x_data[-1], y_data[-1], 'ro', markersize=10, label='Current Waypoint')
    
    ax_wp.legend()

    # --- Save the Figure ---
    filename = os.path.join(OUTPUT_DIR, f'frame_{frame_count:05d}.png')
    fig.savefig(filename)
    rospy.loginfo(f"Saved figure: {filename}")


def main():
    """
    Main function to initialize the ROS node and the saving loop.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        rospy.loginfo(f"Created output directory: {OUTPUT_DIR}")

    rospy.init_node('camera_waypoint_visualizer', anonymous=True)

    rospy.loginfo("Visualizer node started. Will save a figure every second.")
    rospy.loginfo(f"Subscribing to camera topic: {CAMERA_TOPIC}")
    rospy.loginfo(f"Subscribing to waypoint topic: {WAYPOINT_TOPIC}")

    # --- Subscribers ---
    rospy.Subscriber(CAMERA_TOPIC, Image, camera_callback, queue_size=1)
    rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, waypoint_callback, queue_size=10)
    
    # --- Matplotlib Figure Setup ---
    fig, (ax_img, ax_wp) = plt.subplots(1, 2, figsize=(15, 7))
    
    # --- Main Saving Loop ---
    rate = rospy.Rate(1)  # 1 Hz (once per second)
    frame_count = 0
    while not rospy.is_shutdown():
        update_and_save_figure(fig, ax_img, ax_wp, frame_count)
        frame_count += 1
        rate.sleep()

    # Close the plot window when the script is interrupted
    plt.close(fig)
    rospy.loginfo("Shutting down visualizer.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

