import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import Point32
from math import sin, cos
import numpy as np
from sklearn.cluster import DBSCAN
from example_interfaces.msg import Int64

# Global Variables
MOVING_OBSTACLE_THRESHOLD = 0.5

DBSCAN_EPS = 0.69
DBSCAN_MIN_SAMPLES = 8

LAST_TIME_UPDATE = 40
TRACKING_THRESHHOLD = 0.90
dt = 0.1

class People():
    """
    Class to represent a tracked person.

    Attributes:
        position (tuple): Current position of the person (x, y).
        velocity (tuple): Current velocity of the person (vx, vy).
        updated (bool): Flag indicating if the person's state has been updated.
        lifetime (int): Remaining time steps before considering the person as lost.
    """
    def __init__(self, initial_position: tuple):
        self.position = initial_position
        self.velocity = (0.0,0.0)
        self.updated = True
        self.lifetime = LAST_TIME_UPDATE

    def predict(self):
        """
        Predict the next position based on the current state.

        Returns:
            tuple: Predicted next position (x, y).
        """
        next_estimate = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])
        return next_estimate

    def update(self, current_observation: tuple):
        """
        Update the person's state based on a new observation.

        Args:
            current_observation (tuple): New observation of the person's position (x, y).
        """
        self.velocity = (current_observation[0] - self.position[0], current_observation[1] - self.position[1])
        self.position = current_observation


class DetectionAndTracking(Node):
    """
    Node for detecting and tracking moving objects.

    This node subscribes to LaserScan messages, processes the data to identify moving obstacles,
    performs DBSCAN clustering to find centroids, and tracks the centroids over time.

    Attributes:
        unique_people_count (int): Total count of unique people.
        tracked_people (list): List of tracked People objects.
        time_step (int): Current time step.
        next_id (int): Next unique ID for a tracked person.
    """
    def __init__(self):
        super().__init__('detection_and_tracking')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.LaserScanCallback, 10)
        self.publisher_moving = self.create_publisher(PointCloud, '/point_cloud_moving', 10)
        self.publisher_stationary = self.create_publisher(PointCloud, '/point_cloud_stationary', 10)
        self.publisher_centroid = self.create_publisher(PointCloud, '/person_locations', 10)

        self.publisher_count = self.create_publisher(Int64, '/person_count', 10)
        self.first_scan_data = None
        self.header = None
        self.unique_people_count = 0
        self.tracked_people = []
        self.time_step = 0
        self.next_id = 1

    def LaserScanCallback(self, scan: LaserScan):
        """
        Callback function for processing LaserScan messages.

        Args:
            scan (sensor_msgs.msg.LaserScan): The incoming LaserScan message.
        """
        if self.first_scan_data is None:
            self.first_scan_data = scan.ranges
            self.header = scan.header
            return

        moving_points, stationary_points = self.identify_moving_obstacles(scan)
        self.publisher_moving.publish(moving_points)
        self.publisher_stationary.publish(stationary_points)

        identified_clusters, centroids = self.dbscan_clustering(moving_points)
        self.publisher_centroid.publish(centroids)

        self.time_step += 1

        current_centroids = [(p.x, p.y) for p in centroids.points]

        #prediction
        predicted_positions = [track.predict() for track in self.tracked_people]

        used_centroids = []
        for i, predicted_position in enumerate(predicted_positions):
            min_dist = float('inf')
            best_match = None
            for j, centroid in enumerate(current_centroids):
                dist = self.distance_2d(predicted_position, centroid)
                if dist < min_dist and j not in used_centroids:
                    min_dist = dist
                    best_match = j

                if best_match is not None and min_dist < TRACKING_THRESHHOLD:
                    self.tracked_people[i].update(current_centroids[best_match])
                    used_centroids.append(best_match)

        for i, centroid in enumerate(current_centroids):
            if i not in used_centroids:
                self.tracked_people.append(People(centroid))
                self.unique_people_count += 1

        for track in self.tracked_people:
            if not track.updated:
                track.lifetime -= 1

        # Remove tracks with lifetime <= 0
        self.tracked_people = [track for track in self.tracked_people if track.lifetime > 0]

        self.get_logger().info(f"Total unique people counted: {self.unique_people_count}")
        count_msg = Int64()
        count_msg.data = self.unique_people_count
        self.publisher_count.publish(count_msg)

    def distance_2d(self, point1, point2):
        """
        Calculate 2D Euclidean distance between two points.

        Args:
            point1 (tuple): (x, y) coordinates of the first point.
            point2 (tuple): (x, y) coordinates of the second point.

        Returns:
            float: The 2D Euclidean distance between the points.
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def identify_moving_obstacles(self, scan: LaserScan):
        """
        Identify moving obstacles from the LaserScan data.

        Args:
            scan (sensor_msgs.msg.LaserScan): The LaserScan message.

        Returns:
            sensor_msgs.msg.PointCloud: Point cloud of moving obstacles.
            sensor_msgs.msg.PointCloud: Point cloud of stationary obstacles.
        """
        moving_points = []
        stationary_points = []

        for i, current_range in enumerate(scan.ranges):
            static_range = self.first_scan_data[i]

            if current_range < (static_range - MOVING_OBSTACLE_THRESHOLD):
                bearing_i = scan.angle_min + i * scan.angle_increment
                moving_points.append(Point32(
                    x = current_range * cos(bearing_i),
                    y = current_range * sin(bearing_i),
                    z = 0.0
                ))
            else:
                bearing_i = scan.angle_min + i * scan.angle_increment
                stationary_points.append(Point32(
                    x = current_range * cos(bearing_i),
                    y = current_range * sin(bearing_i),
                    z = 0.0
                ))

        mov = PointCloud(header=scan.header, points=moving_points)
        stat = PointCloud(header=scan.header, points=stationary_points)

        return mov, stat

    def dbscan_clustering(self, moving_points: PointCloud):
        """
        Perform DBSCAN clustering on the moving points.

        Args:
            moving_points (sensor_msgs.msg.PointCloud): Point cloud of moving obstacles.

        Returns:
            list: List of clusters identified by DBSCAN.
            sensor_msgs.msg.PointCloud: Point cloud of centroids of moving obstacles.
        """
        points = np.array([[p.x, p.y] for p in moving_points.points])

        if not points.any():
            return [], PointCloud(header=moving_points.header, points=[])

        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
        labels = clustering.labels_

        clusters = [[] for _ in range(max(labels) + 1)]
        centroids = []

        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(points[i])

        for cluster_points in clusters:
            if len(cluster_points) >= DBSCAN_MIN_SAMPLES:
                cluster_points = np.array(cluster_points)
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(Point32(x=centroid[0], y=centroid[1], z=0.0))

        centroids_data = PointCloud(header=moving_points.header, points=centroids)

        return clusters, centroids_data

def main(args = None):
    rclpy.init(args = args)
    node = DetectionAndTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
