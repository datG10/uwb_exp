FROM osrf/ros:humble-desktop

# Metadata for the Docker image
LABEL maintainer="demetrosaschalew12@gmail.com"
LABEL version="0.1"
LABEL description="Docker image for UWB experiment."

# Use Bash for executing commands in Dockerfile
SHELL ["/bin/bash", "-c"]

# Install necessary packages
RUN mkdir -p ~/ros2_ws/src && \
    git clone https://github.com/datG10/uwb_exp_indoor.git ~/ros2_ws/src/uwb_exp_indoor

RUN apt-get update && \
    apt-get install -y git python3-pip python3-colcon-common-extensions cmake && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

RUN mkdir -p ~/ros2_ws/uros_ws/src && \
    cd ~/ros2_ws/uros_ws/src && \
    git clone -b humble https://github.com/micro-ROS/micro_ros_setup.git

# Create vicon_ws directory and clone the repository
RUN mkdir -p ~/ros2_ws/vicon_ws/src && \
    cd ~/ros2_ws/vicon_ws/src && \
    git clone https://github.com/OPT4SMART/ros2-vicon-receiver.git && \
    cd ros2-vicon-receiver && \
    ./install_libs.sh 

# Update rosdep and install dependencies
RUN cd ~/ros2_ws/uros_ws && \
    source /opt/ros/humble/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    # Build the micro-ROS workspace
    colcon build --symlink-install && \
    # Source the setup files
    source install/local_setup.bash && \
    # Run the micro-ROS agent setup
    ros2 run micro_ros_setup create_agent_ws.sh && \
    ros2 run micro_ros_setup build_agent.sh

RUN cd ~/ros2_ws/vicon_ws && \
    source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install && \
    source install/local_setup.bash 
# Source ROS, build the workspace, and handle errors if any
RUN source /opt/ros/humble/setup.bash && \
    cd ~/ros2_ws && \
    colcon build --symlink-install || (echo "Failed to build ros2 workspace" && exit 1)

# Final message to indicate setup completion
RUN echo "Setup complete"


