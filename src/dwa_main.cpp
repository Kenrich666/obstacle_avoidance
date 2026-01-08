/**
 * @file main.cpp
 * @brief DWA Planner Node Entry Point
 */

#include <ros/ros.h>
#include "custom_dwa_planner/dwa_ros_node.h"
#include <iostream> // 新增

int main(int argc, char** argv) {
    ros::init(argc, argv, "custom_dwa_planner");

    try {
        custom_dwa_planner::DWAROSNode node;
        node.run();
    } catch (const std::exception& e) {
        // 使用 std::cerr 替代 ROS_ERROR 以防止日志未刷新
        std::cerr << "\n\n[FATAL ERROR] Node Crashed: " << e.what() << "\n\n" << std::endl;
        ROS_ERROR("Unhandled exception: %s", e.what());
        return 1;
    }

    return 0;
}