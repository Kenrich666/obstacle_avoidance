cmake_minimum_required(VERSION 3.0.2)
project(obstacle_avoidance)

add_compile_options(-std=c++11)

# 1. 查找依赖包
# 必须包含 roscpp, sensor_msgs, marine_acoustic_msgs
find_package(catkin REQUIRED COMPONENTS
  roscpp                # 新增: C++ 核心
  sensor_msgs           # 新增: PointCloud2
  marine_acoustic_msgs  # 新增: ProjectedSonarImage
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
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES obstacle_avoidance
#  CATKIN_DEPENDS roscpp sensor_msgs ...
#  DEPENDS system_lib
)

# 2. 包含头文件目录
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 3. 声明 C++ 可执行文件
# 假设源文件在 src/sonar_to_cloud_node.cpp
add_executable(sonar_to_cloud_node src/sonar_to_cloud_node.cpp)
# 4. 链接库
target_link_libraries(sonar_to_cloud_node
  ${catkin_LIBRARIES}
)

# 5. 添加依赖关系
# 确保在编译此节点前，相关的消息头文件已经生成
add_dependencies(sonar_to_cloud_node ${catkin_EXPORTED_TARGETS})