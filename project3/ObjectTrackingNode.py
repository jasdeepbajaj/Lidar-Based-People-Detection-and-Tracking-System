import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
import numpy as np
from example_interfaces.msg import Int64


LAST_TIME_UPDATE = 20
TRACKING_THRESHHOLD = 1.0 #1.0

dt = 0.1

class People():
    def __init__(self, initial_position: tuple):
        self.position = initial_position
        self.velocity = (0.0,0.0)
        self.updated = True
        self.lifetime = LAST_TIME_UPDATE

    def predict(self):
        next_estimate = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])
        return next_estimate

    def update(self, current_observation: tuple):
        self.velocity = (current_observation[0] - self.position[0], current_observation[1] - self.position[1])
        self.position = current_observation

class ObjectTrackingNode(Node):
    def __init__(self):
        super().__init__('object_tracking_node')
        self.subscription = self.create_subscription(PointCloud, '/person_location', self.TrackerCallback, 10)
        self.publisher_count = self.create_publisher(Int64, '/person_count', 10)
        self.first_scan_data = None
        self.header = None

        self.unique_people_count = 0
        self.tracked_people = []

    def TrackerCallback(self, centroids: PointCloud):
        current_centroids = [(p.x, p.y) for p in centroids.points]

        #prediction
        predicted_positon = [track.predict() for track in self.tracked_people]


        used_centroids = []
        for i, predicted_pos in enumerate(predicted_positon):
            min_dist = float('inf')
            best_match = None
            for j, centroid in enumerate(current_centroids):
                dist = self.distance_2d(predicted_pos, centroid)
                if dist < min_dist and j not in used_centroids:
                    min_dist = dist
                    best_match = j

                if best_match is not None and min_dist < TRACKING_THRESHHOLD:  # Distance threshold can be adjusted
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
        self.tracks = [track for track in self.tracked_people if track.lifetime > 0]

        self.get_logger().info(f"Total unique people counted: {self.unique_people_count}")
        count_msg = Int64()
        count_msg.data = self.unique_people_count
        self.publisher_count.publish(count_msg)

    def distance_2d(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main(args = None):
    rclpy.init(args = args)
    node = ObjectTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
