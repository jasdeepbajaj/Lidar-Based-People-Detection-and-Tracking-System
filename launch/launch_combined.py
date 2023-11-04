from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command

def generate_launch_description():

    node1 = Node(package='project3', executable='DetectionAndTracking')

    arg = DeclareLaunchArgument('arg_name')

    bag_file_path = Command(["echo /home/jasdeep/ros2_ws/src/project3/bags/bags/", LaunchConfiguration('arg_name')])
    bag_record_file_path = Command(["echo /home/jasdeep/ros2_ws/src/project3/bags/recordings/", LaunchConfiguration('arg_name'),"-out"])

    ep = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_file_path],
        shell=True,
        name='ros2_bag_play'
    )

    ep2 = ExecuteProcess(
        cmd=['ros2', 'bag', 'record','/scan','/person_locations','/person_count','-o', bag_record_file_path],
        shell=True,
        name='ros2_bag_record'
    )

    ld = LaunchDescription([
        arg,
        node1,
        ep,
        ep2,
    ])

    return ld
