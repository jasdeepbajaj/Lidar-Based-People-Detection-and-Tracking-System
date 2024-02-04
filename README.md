# Object Detection and Tracking ROS2 Package

## Overview
This project aims to develop software that processes data from a LIDAR sensor to detect and track the movement of people in a given scene. 

## Task Description
The project provides a collection of bag files recorded by a robot equipped with a laser range finder (LIDAR). Each bag file contains data where the robot and nearby obstacles remain stationary, while one or more people walk past the robot. The task involves:

- Creating a system to identify and track visible people.
- Publishing a `sensor_msgs/PointCloud` message on the topic `/person_locations`, containing one point for each visible person at the estimated center location.
- Publishing an `example_interfaces/Int64` message on the topic `/person_count`, indicating the total number of unique people observed.

## Dependencies
- ROS2 (Humble Hawksbill)
- Python (3.10)
- NumPy
- scikit-learn

## Installation
1. Clone this repository into your ROS2 workspace:
    ```bash
    https://github.com/jasdeepbajaj/project3.git
    ```
2. Build the package:
    ```bash
    colcon build --packages-select project3
    ```
### Parameters and Tuning
Movement Detection Threshold, Euclidean Clustering Parameters (distance tolerance, min/max cluster size), and Tracking Update Interval were empirically determined for optimal performance.
Results and Expectations
The system demonstrated high accuracy in detecting and tracking people in various test scenarios. Fine-tuning parameters allowed for significant improvements, especially in challenging conditions like crowded environments or at the edges of the LIDAR range.
   
## Launch File Usage

A launch file is provided to start both nodes and play a bag file for testing. To use the launch file:
    ```bash
    ros2 launch project3 launch.py arg_name:=example9
    ```

## Launch File Details

The launch file object_detection_and_tracking_launch.py performs the following actions:

1.    Launches the ObjectDetectionNode.
2.    Launches the ObjectTrackingNode.
3.    Plays a ROS2 bag file containing laser scan data.

## Bag Files

Sample bag files for testing the package are available in the bags directory.
