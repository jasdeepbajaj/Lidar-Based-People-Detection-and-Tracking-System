import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import Point32
from math import sin, cos
import numpy as np
from sklearn.cluster import DBSCAN

# Global Variables
MOVING_OBSTACLE_THRESHOLD = 0.5

DBSCAN_EPS = 0.69 #0.69
DBSCAN_MIN_SAMPLES = 10

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.LaserScanCallback, 10)
        self.publisher_moving = self.create_publisher(PointCloud, '/point_cloud_moving', 10)
        self.publisher_stationary = self.create_publisher(PointCloud, '/point_cloud_stationary', 10)
        self.publisher_centroid = self.create_publisher(PointCloud, '/person_location', 10)
        self.first_scan_data = None
        

        self.unique_people_count = 0
        self.tracked_people = []

        self.next_id = 1

    def LaserScanCallback(self, scan: LaserScan):
        if self.first_scan_data is None:
            self.first_scan_data = scan.ranges
            return

        moving_points, stationary_points = self.identify_moving_obstacles(scan)
        self.publisher_moving.publish(moving_points)
        self.publisher_stationary.publish(stationary_points)

        identified_clusters, centroids = self.dbscan_clustering(moving_points)
        self.publisher_centroid.publish(centroids)

    def distance_2d(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def identify_moving_obstacles(self, scan: LaserScan):
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
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
