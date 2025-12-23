cmake_minimum_required(VERSION 3.0.2)
project(obstacle_avoidance)

add_compile_options(-std=c++14) # 建议升级到 C++14 以获得更好的 PCL/Eigen 支持，C++11 也可以

# 1. 查找依赖包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  marine_acoustic_msgs
  gazebo_msgs
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  uuv_control_msgs
  uuv_gazebo_ros_plugins_msgs
  uuv_trajectory_control
  uuv_world_ros_plugins_msgs
  visualization_msgs
  # --- 新增依赖 (用于 sonar_occupancy_mapper) ---
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

# --- 查找系统库 ---
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obstacle_avoidance
#  CATKIN_DEPENDS roscpp sensor_msgs ...
#  DEPENDS system_lib
)

# 2. 包含头文件目录
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}     # 新增: PCL 头文件
  ${Eigen3_INCLUDE_DIRS}  # 新增: Eigen 头文件
)

# ---------------------------------------------------------
# 原有的节点 (保持不变)
# ---------------------------------------------------------
add_executable(sonar_to_cloud_node src/sonar_to_cloud_node.cpp)
target_link_libraries(sonar_to_cloud_node
  ${catkin_LIBRARIES}
)
add_dependencies(sonar_to_cloud_node ${catkin_EXPORTED_TARGETS})

# ---------------------------------------------------------
# 新增的节点 (Sonar Occupancy Mapper)
# ---------------------------------------------------------
add_executable(sonar_occupancy_mapper src/sonar_occupancy_mapper.cpp)

target_link_libraries(sonar_occupancy_mapper
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}       # 必须链接 PCL 库
)

add_dependencies(sonar_occupancy_mapper ${catkin_EXPORTED_TARGETS})