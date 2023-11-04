from launch import LaunchDescription  # Importing LaunchDescription class
from launch_ros.actions import Node  # Importing Node action
from launch.actions import DeclareLaunchArgument, ExecuteProcess  # Importing LaunchArgument and ExecuteProcess actions
from launch.substitutions import LaunchConfiguration, Command  # Importing LaunchConfiguration and Command substitutions

def generate_launch_description():
    # Define a launch description generator function

    node1 = Node(package='project3', executable='ObjectDetectionNode')  
    # Create a Node with package 'project3' and executable 'ObjectDetectionNode'

    node2 = Node(package='project3', executable='ObjectTrackingNode')  
    # Create a second Node with package 'project3' and executable 'ObjectTrackingNode'

    arg = DeclareLaunchArgument('arg_name')  
    # Declare a launch argument named 'arg_name'

    bag_file_path = Command(["echo /home/jasdeep/ros2_ws/src/project3/bags/bags/", LaunchConfiguration('arg_name')])  
    # Define a command substitution to set the bag file path

    bag_record_file_path = Command(["echo /home/jasdeep/ros2_ws/src/project3/bags/recordings/", LaunchConfiguration('arg_name'),"-out"])  
    # Define a command substitution to set the bag record file path

    ep = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_file_path],  
        # Define an ExecuteProcess action to play a bag file
        shell=True,
        name='ros2_bag_play'  
        # Set the name of the process
    )

    ep2 = ExecuteProcess(
        cmd=['ros2', 'bag', 'record','/scan','/person_location','/person_count','-o', bag_record_file_path],  
        # Define an ExecuteProcess action to record specific topics
        shell=True,
        name='ros2_bag_record'  
        # Set the name of the process
    )

    ld = LaunchDescription([
        arg,  # Add the launch argument to the launch description
        node1,  # Add the first Node action to the launch description
        node2,  # Add the second Node action to the launch description
        ep,  # Add the ExecuteProcess action to the launch description
        ep2,  # Add the second ExecuteProcess action to the launch description
    ])

    return ld  # Return the generated launch description
