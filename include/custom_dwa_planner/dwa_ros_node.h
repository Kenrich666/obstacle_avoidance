#ifndef CUSTOM_DWA_PLANNER_DWA_ROS_NODE_H
#define CUSTOM_DWA_PLANNER_DWA_ROS_NODE_H

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Twist.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
// 包含核心算法和配置头文件
#include "custom_dwa_planner/dwa_core.h"
#include "custom_dwa_planner/dwa_config.h"

namespace custom_dwa_planner {

/**
 * @brief DWA 算法的 ROS 封装节点类
 * 负责处理 ROS 消息收发、TF 转换和可视化，将数据传递给 DWACore 进行计算。
 */
class DWAROSNode {
public:
    DWAROSNode();
    ~DWAROSNode() = default;

    /**
     * @brief 主循环函数，包含 spin 和控制频率控制
     */
    void run();

private:
    // --- ROS 回调函数 ---
    
    /**
     * @brief 全局路径回调
     * 接收来自上层规划器 (如 Navfn/GlobalPlanner) 的路径
     */
    void pathCallback(const nav_msgs::Path::ConstPtr& msg);

    /**
     * @brief 里程计回调
     * 更新机器人当前位姿和速度
     */
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);

    /**
     * @brief 代价地图回调
     * 接收局部或全局代价地图用于避障
     */
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);

    // --- 辅助功能函数 ---

    /**
     * @brief 从参数服务器加载参数
     */
    void loadParams();

    /**
     * @brief 检查是否长时间未移动 (堵塞检测)
     * @return true 如果超时堵塞
     */
    bool checkStuck();

    /**
     * @brief 将全局路径转换到控制坐标系 (通常是 world 或 odom)
     * @return 转换后的路径指针，失败返回 nullptr
     */
    nav_msgs::Path::Ptr transformGlobalPlan();

    /**
     * @brief 在局部路径上搜索目标点 (胡萝卜)
     * @param path_msg 局部路径
     * @param target [输出] 目标点坐标
     * @return true 成功找到, false 失败或到达终点
     */
    bool getLocalTarget(const nav_msgs::Path::Ptr& path_msg, Eigen::Vector2d& target);

    // --- 可视化函数 ---
    
    void visualizeTarget(const Eigen::Vector2d& target);
    void visualizeTrajectories(const std::vector<Trajectory>& all_trajs, const Trajectory& best_traj);

private:
    // ROS 句柄
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // 订阅者
    ros::Subscriber path_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber map_sub_;
    
    // 发布者
    ros::Publisher cmd_pub_;
    ros::Publisher traj_pub_;      // 发布所有候选轨迹簇
    ros::Publisher best_traj_pub_; // 发布选中的最佳轨迹
    ros::Publisher target_pub_;    // 发布当前的追踪目标点

    // TF 变换工具
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // 核心算法对象
    DWACore core_;
    DWAConfig config_;

    // --- 运行时状态数据 ---
    
    State current_state_;
    
    // 原始全局路径 (Map Frame)
    nav_msgs::Path::ConstPtr raw_global_path_;
    
    // 持有当前地图消息的智能指针，确保数据内存有效
    nav_msgs::OccupancyGrid::ConstPtr current_map_msg_;
    
    // 机器人深度 (用于水下可视化)
    double robot_z_ = -76.0; 
    
    // 状态标志
    bool odom_received_ = false;

    // 路径跟随状态变量
    int last_path_index_ = 0; // 滑动窗口索引
    ros::Time last_valid_move_time_; // 堵塞计时器
};

} // namespace custom_dwa_planner

#endif // CUSTOM_DWA_PLANNER_DWA_ROS_NODE_H