cmake_minimum_required(VERSION 3.10)
project(kitti_publisher_conf)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT paths
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-10.12.0.36/include)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-10.12.0.36/lib)

# CUDA
find_package(CUDA REQUIRED)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosbag2_cpp REQUIRED)
# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  /usr/include/x86_64-linux-gnu
)

# Link directories
link_directories(
  ${TENSORRT_LIBRARY_DIR}
)

# Executable
add_executable(kitti_publisher_conf_cuda_node src/kitti_publisher_conf_cuda_node.cpp)

# Dependencies
ament_target_dependencies(kitti_publisher_conf_cuda_node
  rclcpp
  sensor_msgs
  OpenCV
  rosbag2_cpp
  rosidl_typesupport_cpp
  geometry_msgs
  tf2_ros
  cv_bridge
)

# Link libraries
target_link_libraries(kitti_publisher_conf_cuda_node
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  nvinfer
  nvonnxparser
  nvinfer_plugin
)

# Install the node executable
install(TARGETS
  kitti_publisher_conf_cuda_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)


ament_package()
