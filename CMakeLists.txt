cmake_minimum_required(VERSION 3.0.2)
project(obstacle_avoidance)

# 使用 C++14 标准
add_compile_options(-std=c++14)

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
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES obstacle_avoidance
#  CATKIN_DEPENDS roscpp sensor_msgs ...
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

# --- 现有的节点 ---
add_executable(sonar_to_cloud_node src/sonar_to_cloud_node.cpp)
target_link_libraries(sonar_to_cloud_node
  ${catkin_LIBRARIES}
)
add_dependencies(sonar_to_cloud_node ${catkin_EXPORTED_TARGETS})

add_executable(sonar_occupancy_mapper src/sonar_occupancy_mapper.cpp)
target_link_libraries(sonar_occupancy_mapper
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
add_dependencies(sonar_occupancy_mapper ${catkin_EXPORTED_TARGETS})

# --- [新增] DWA Planner 节点 ---
add_executable(custom_dwa_planner_node
  src/dwa_core.cpp
  src/dwa_ros_node.cpp
  src/dwa_main.cpp
)

# 注意：如果你的 main.cpp 名字不同（例如为了避免冲突改名为 dwa_main.cpp），请修改上面的文件名
# 建议：如果 src/main.cpp 是专为 DWA 写的，这没问题。

target_link_libraries(custom_dwa_planner_node
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES} 
)