import rclpy
import time
import csv
import datetime
import threading
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from vicon_receiver.msg import Position
import math

class UWB_Exiprement(Node):
    def __init__(self):
        super().__init__("UWB_Exiprement")

        # Initialize position data (Vicon) and distance data (UWB)
        self.tag_position = np.array([math.nan, math.nan, math.nan])  # Tag position from Vicon
        self.anchor_position = np.array([math.nan, math.nan, math.nan])  # Anchor position from Vicon
        self.distance_data = np.array([math.nan, math.nan, math.nan])  # [True Distance, Measured Distance, dBm]

        self.data_ready = {'tag': False, 'anchor': False, 'distance': False}
        self.recording = False
        self.experiment_count = 0
        self.iteration_count = 0
        self.max_iterations = 5 
        self.delay_between_iterations = 0.5  
        self.experiment_data = []

        # Create the initial CSV file and write the header
        with open('dataset.csv', 'w', newline='') as file:
            self.writer = csv.writer(file)
            header = ["experiment", "datetime", "elapsed_time", "tag_x", "tag_y", "tag_z", "anc_x", "anc_y", "anc_z", "anchor_id", "dis_mes", "dbm"]
            self.writer.writerow(header)

        # Subscribers for Vicon position and UWB distances
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1)

        self.sub_pos_tag = self.create_subscription(Position, "/vicon/tag/tag", self.pos_callback_tag, qos_policy)
        self.sub_pos_anc = self.create_subscription(Position, "/vicon/anc/anc", self.pos_callback_anchor, qos_policy)
        self.sub_uwb = self.create_subscription(Float32MultiArray, 'uwb_distance', self.dis_callback, qos_policy)

        self.prev_time = time.time()

    def pos_callback_tag(self, msg: Position):
        # Process position data from Vicon system
        self.tag_position = np.array([msg.x_trans / 1000, msg.y_trans / 1000, msg.z_trans / 1000])
        self.data_ready['tag'] = True
        self.log_data()

    def pos_callback_anchor(self, msg: Position):
        # Process position data from Vicon system
        self.anchor_position = np.array([msg.x_trans / 1000, msg.y_trans / 1000, msg.z_trans / 1000])
        self.data_ready['anchor'] = True
        self.log_data()

    def dis_callback(self, msg: Float32MultiArray):
        # Process UWB distance data
        if len(msg.data) >= 3:
            self.distance_data = np.array([msg.data[0], msg.data[1], msg.data[2]])
            self.data_ready['distance'] = True
            self.log_data()

    def log_data(self):
        # Log data continuously
        self.get_logger().info(f'Tag position: {self.tag_position}, Anchor position: {self.anchor_position}, Distance data: {self.distance_data}')

        # If recording, save the data
        if self.recording:
            self.save_data()

    def save_data(self):
        date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = time.time() - self.prev_time

        # Save the current values (with NaNs if data is missing)
        row = [
            self.experiment_count,
            date_time,
            elapsed_time,
            *self.tag_position,
            *self.anchor_position,
            *self.distance_data
        ]

        # Append the data to the experiment data list
        self.experiment_data.append(row)

        # Delay between iterations
        time.sleep(self.delay_between_iterations)

        # Update the previous time
        self.prev_time = time.time()

        # Increment iteration count and stop after reaching the maximum number of iterations
        self.iteration_count += 1
        if self.iteration_count >= self.max_iterations:
            self.stop_recording()

    def start_recording(self):
        self.get_logger().info(f'Starting experiment {self.experiment_count + 1}...')
        self.recording = True
        self.iteration_count = 0
        self.experiment_data = []
        self.prev_time = time.time()

    def stop_recording(self):
        self.get_logger().info(f'Stopping experiment {self.experiment_count + 1}...')
        self.recording = False

        # Save the experiment data to the CSV file
        with open('dataset_outdoor.csv', 'a', newline='') as file:
            self.writer = csv.writer(file)
            self.writer.writerows(self.experiment_data)

        self.experiment_count += 1

def main(args=None):
    rclpy.init(args=args)
    observation_node = UWB_Exiprement()

    # Thread to listen for key presses to start/stop recording
    def experiment_control():
        while True:
            command = input("Press 's' to start an experiment, or 'q' to quit:\n")
            if command.lower() == 's':
                if not observation_node.recording:
                    observation_node.start_recording()
                else:
                    print("Already recording an experiment. Please wait for it to complete.")
            elif command.lower() == 'q':
                if observation_node.recording:
                    observation_node.stop_recording()
                observation_node.get_logger().info("Quitting...")
                if rclpy.ok():  # Check if rclpy is still running before shutting down
                    rclpy.shutdown()
                break

    control_thread = threading.Thread(target=experiment_control)
    control_thread.start()

    try:
        rclpy.spin(observation_node)
    except KeyboardInterrupt:
        print("Terminating Node...")
    finally:
        observation_node.destroy_node()
        if rclpy.ok():  # Ensure rclpy is still running before shutting down
            rclpy.shutdown()

if __name__ == '__main__':
    main()
