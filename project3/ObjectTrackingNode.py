import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
import numpy as np
from example_interfaces.msg import Int64

LAST_TIME_UPDATE = 20
TRACKING_THRESHHOLD = 1.0
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

class ObjectTrackingNode(Node):
    """
    Node for object tracking using PointCloud data.

    This node subscribes to PointCloud messages containing centroid information, 
    performs tracking of people, and publishes the count of unique people.

    Attributes:
        unique_people_count (int): Total count of unique people.
        tracked_people (list): List of tracked People objects.
    """
    def __init__(self):
        super().__init__('object_tracking_node')
        self.subscription = self.create_subscription(PointCloud, '/person_location', self.TrackerCallback, 10)
        self.publisher_count = self.create_publisher(Int64, '/person_count', 10)
        self.first_scan_data = None
        self.header = None
        self.unique_people_count = 0
        self.tracked_people = []

    def TrackerCallback(self, centroids: PointCloud):
        """
        Callback function for processing PointCloud messages.

        Args:
            centroids (sensor_msgs.msg.PointCloud): The incoming PointCloud message.
        """
        current_centroids = [(p.x, p.y) for p in centroids.points]

        # Prediction
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

def main(args = None):
    rclpy.init(args = args)
    node = ObjectTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
