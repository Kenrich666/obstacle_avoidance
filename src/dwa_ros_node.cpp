#include "custom_dwa_planner/dwa_ros_node.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>

namespace custom_dwa_planner
{

    DWAROSNode::DWAROSNode() : private_nh_("~"), tf_listener_(tf_buffer_)
    {
        // 1. 加载参数
        loadParams();

        // [新增] 初始化评分日志
        // 路径设为 /tmp/dwa_scoring_log.csv，方便查看
        core_.initDebugLog("/tmp/dwa_scoring_log.csv");
        ROS_INFO("DWA scoring log initialized at /tmp/dwa_scoring_log.csv");

        // 2. 初始化通信接口
        // Subscriber
        path_sub_ = nh_.subscribe("/move_base/NavfnROS/plan", 1, &DWAROSNode::pathCallback, this);
        odom_sub_ = nh_.subscribe("/rexrov2/pose_gt", 1, &DWAROSNode::odomCallback, this);
        map_sub_ = nh_.subscribe("/projected_sonar_map", 1, &DWAROSNode::mapCallback, this);

        // Publisher
        cmd_pub_ = nh_.advertise<geometry_msgs::Twist>("/rexrov2/cmd_vel_move_base", 1);
        traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/dwa_trajectories", 1);
        best_traj_pub_ = nh_.advertise<nav_msgs::Path>("/dwa_best_traj", 1);
        target_pub_ = nh_.advertise<visualization_msgs::Marker>("/dwa_current_target", 1);

        // 初始化状态
        last_valid_move_time_ = ros::Time::now();

        // ROS_INFO("DWA路径规划器节点初始化完成。");
        ROS_INFO("DWA Planner Node Initialized.");
    }

    void DWAROSNode::loadParams()
    {
        DWAConfig cfg;

        // 1. 基础参数加载
        private_nh_.param("max_vel_x", cfg.max_vel_x, cfg.max_vel_x);
        private_nh_.param("min_vel_x", cfg.min_vel_x, cfg.min_vel_x);
        private_nh_.param("max_rot_vel", cfg.max_rot_vel, cfg.max_rot_vel);

        private_nh_.param("acc_lim_x", cfg.acc_lim_x, cfg.acc_lim_x);
        private_nh_.param("acc_lim_theta", cfg.acc_lim_theta, cfg.acc_lim_theta);

        private_nh_.param("sim_time", cfg.sim_time, cfg.sim_time);
        private_nh_.param("sim_dt", cfg.sim_dt, cfg.sim_dt);

        private_nh_.param("path_distance_bias", cfg.path_distance_bias, cfg.path_distance_bias);
        private_nh_.param("goal_distance_bias", cfg.goal_distance_bias, cfg.goal_distance_bias);
        private_nh_.param("occdist_scale", cfg.occdist_scale, cfg.occdist_scale);

        private_nh_.param("lookahead_dist", cfg.lookahead_dist, cfg.lookahead_dist);
        private_nh_.param("goal_reach_dist", cfg.goal_reach_dist, cfg.goal_reach_dist);
        private_nh_.param("stuck_timeout", cfg.stuck_timeout, cfg.stuck_timeout);

        // 2. 解析 Footprint (移除 try-catch，使用安全检查)
        XmlRpc::XmlRpcValue footprint_list;
        if (private_nh_.getParam("footprint", footprint_list))
        {
            if (footprint_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
            {
                // 只有当参数确实是数组时才清理默认值
                cfg.footprint.clear();

                for (int i = 0; i < footprint_list.size(); ++i)
                {
                    XmlRpc::XmlRpcValue point = footprint_list[i];

                    // 检查内层是否为数组且长度为2
                    if (point.getType() == XmlRpc::XmlRpcValue::TypeArray && point.size() == 2)
                    {
                        double x = 0.0;
                        double y = 0.0;

                        // 安全提取 X (兼容 int 和 double)
                        if (point[0].getType() == XmlRpc::XmlRpcValue::TypeInt)
                            x = static_cast<int>(point[0]);
                        else if (point[0].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                            x = static_cast<double>(point[0]);

                        // 安全提取 Y (兼容 int 和 double)
                        if (point[1].getType() == XmlRpc::XmlRpcValue::TypeInt)
                            y = static_cast<int>(point[1]);
                        else if (point[1].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                            y = static_cast<double>(point[1]);

                        cfg.footprint.push_back(Eigen::Vector2d(x, y));
                    }
                }
            }
            else
            {
                // ROS_WARN("检测到参数 'footprint' 不是数组类型，将采用默认配置。");
                ROS_WARN("Parameter 'footprint' found but it is not an array. Using default.");
            }
        }

        // 3. 更新核心配置
        core_.updateConfig(cfg);
        config_ = cfg;
    }

    void DWAROSNode::pathCallback(const nav_msgs::Path::ConstPtr &msg)
    {
        if (msg->poses.empty())
        {
            raw_global_path_ = nullptr;
            return;
        }

        raw_global_path_ = msg;
        last_path_index_ = 0;                     // 重置搜索索引
        last_valid_move_time_ = ros::Time::now(); // 重置超时
        ROS_INFO("Received new plan with %lu points, frame: %s", msg->poses.size(), msg->header.frame_id.c_str());
        // ROS_INFO("接收新路径规划结果：路径点数量=%lu，参考坐标系=%s", msg->poses.size(), msg->header.frame_id.c_str());
    }

    void DWAROSNode::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        // 提取状态
        current_state_.x = msg->pose.pose.position.x;
        current_state_.y = msg->pose.pose.position.y;

        // [修复] 动态获取 Z (保留 Python 逻辑: -76.0 default, or from msg)
        // robot_z_ = -76.0;
        robot_z_ = msg->pose.pose.position.z;

        // 提取 Yaw
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        current_state_.yaw = tf2::getYaw(q);

        // 提取速度 (假设 Twist 在 Body/Child Frame)
        current_state_.v = msg->twist.twist.linear.x;
        current_state_.w = msg->twist.twist.angular.z;

        odom_received_ = true;
    }

    void DWAROSNode::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        // 保存智能指针，确保数据所有权
        current_map_msg_ = msg;

        // 更新核心层地图指针
        // 注意：msg->data 是 std::vector<int8_t>
        core_.setMap(msg->data.data(), msg->info.width, msg->info.height,
                     msg->info.resolution, msg->info.origin.position.x, msg->info.origin.position.y);
    }

    bool DWAROSNode::checkStuck()
    {
        if ((ros::Time::now() - last_valid_move_time_).toSec() > config_.stuck_timeout)
        {
            // ROS_WARN("机器人停滞时间过长！正在清除路径规划...");
            ROS_WARN("Robot STUCK for too long! Clearing plan...");
            raw_global_path_ = nullptr; // 强制停止

            geometry_msgs::Twist stop_cmd;
            cmd_pub_.publish(stop_cmd);
            return true;
        }
        return false;
    }

    nav_msgs::Path::Ptr DWAROSNode::transformGlobalPlan()
    {
        if (!raw_global_path_)
            return nullptr;

        std::string target_frame = "world"; // 或者 "odom"

        // 如果已经在目标坐标系，直接返回
        if (raw_global_path_->header.frame_id == target_frame)
        {
            // 创建深拷贝以保证线程安全（虽然这里是单线程）
            nav_msgs::Path::Ptr new_path(new nav_msgs::Path(*raw_global_path_));
            return new_path;
        }

        try
        {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                target_frame, raw_global_path_->header.frame_id, ros::Time(0), ros::Duration(0.5));

            nav_msgs::Path::Ptr new_path(new nav_msgs::Path());
            new_path->header.frame_id = target_frame;
            new_path->header.stamp = ros::Time::now();

            for (const auto &pose : raw_global_path_->poses)
            {
                geometry_msgs::PoseStamped new_pose;
                tf2::doTransform(pose, new_pose, transform);
                new_path->poses.push_back(new_pose);
            }
            return new_path;
        }
        catch (tf2::TransformException &ex)
        {
            // ROS_WARN_THROTTLE(2.0, "TF Transform failed: %s", ex.what());
            // ROS_WARN_THROTTLE(2.0, "坐标变换(TF)执行失败：%s", ex.what());
            ROS_WARN_THROTTLE(2.0, "TF Transform failed: %s", ex.what());
            return nullptr;
        }
    }

    bool DWAROSNode::getLocalTarget(const nav_msgs::Path::Ptr &path_msg, Eigen::Vector2d &target)
    {
        if (!path_msg || path_msg->poses.empty())
            return false;

        double rx = current_state_.x;
        double ry = current_state_.y;
        double min_dist_sq = std::numeric_limits<double>::max();
        int closest_idx = -1;

        // 1. 滑动窗口搜索最近点
        int start_idx = last_path_index_;
        int search_window = 100;
        int end_idx = std::min(start_idx + search_window, (int)path_msg->poses.size());

        for (int i = start_idx; i < end_idx; ++i)
        {
            double px = path_msg->poses[i].pose.position.x;
            double py = path_msg->poses[i].pose.position.y;
            double dist_sq = (px - rx) * (px - rx) + (py - ry) * (py - ry);
            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                closest_idx = i;
            }
        }

        // 2. Fallback 全局搜索 (如果迷失或重置)
        // 25.0 是 5m 的平方
        if (closest_idx == -1 || min_dist_sq > 25.0)
        {
            min_dist_sq = std::numeric_limits<double>::max();
            for (int i = 0; i < (int)path_msg->poses.size(); ++i)
            {
                double px = path_msg->poses[i].pose.position.x;
                double py = path_msg->poses[i].pose.position.y;
                double dist_sq = (px - rx) * (px - rx) + (py - ry) * (py - ry);
                if (dist_sq < min_dist_sq)
                {
                    min_dist_sq = dist_sq;
                    closest_idx = i;
                }
            }
        }

        if (closest_idx != -1)
        {
            last_path_index_ = closest_idx;
        }
        else
        {
            return false;
        }

        // 3. Lookahead (胡萝卜)
        int target_idx = closest_idx;
        double dist_sum = 0.0;
        for (int i = closest_idx; i < (int)path_msg->poses.size() - 1; ++i)
        {
            const auto &p1 = path_msg->poses[i].pose.position;
            const auto &p2 = path_msg->poses[i + 1].pose.position;
            dist_sum += std::hypot(p2.x - p1.x, p2.y - p1.y);

            target_idx = i + 1;
            if (dist_sum > config_.lookahead_dist)
            {
                break;
            }
        }

        const auto &final_pose = path_msg->poses[target_idx].pose.position;
        target.x() = final_pose.x;
        target.y() = final_pose.y;

        // 检查是否到达终点
        const auto &path_end = path_msg->poses.back().pose.position;
        double dist_to_end = std::hypot(rx - path_end.x, ry - path_end.y);
        if (dist_to_end < config_.goal_reach_dist)
        {
            ROS_INFO_THROTTLE(5.0, "Goal Reached!");
            return false; // 到达终点，停止规划
        }

        return true;
    }

    void DWAROSNode::run()
    {
        ros::Rate rate(10); // 10Hz

        while (ros::ok())
        {
            ros::spinOnce();

            // 基础检查
            if (!odom_received_ || !raw_global_path_)
            {
                // 发布零速以防万一
                cmd_pub_.publish(geometry_msgs::Twist());
                rate.sleep();
                continue;
            }

            // 1. 坐标转换
            nav_msgs::Path::Ptr local_plan = transformGlobalPlan();
            if (!local_plan)
            {
                rate.sleep();
                continue;
            }

            // 2. 获取局部目标
            Eigen::Vector2d local_goal;
            if (!getLocalTarget(local_plan, local_goal))
            {
                // 可能是到达终点，或路径无效
                cmd_pub_.publish(geometry_msgs::Twist());
                raw_global_path_ = nullptr; // 重置路径
                rate.sleep();
                continue;
            }

            // 可视化目标
            visualizeTarget(local_goal);

            // 3. DWA 核心计算
            Eigen::Vector2d next_cmd;
            Trajectory best_traj;
            std::vector<Trajectory> all_trajs;

            bool success = core_.computeVelocityCommands(current_state_, local_goal, next_cmd, best_traj, all_trajs);

            // 4. 执行控制
            geometry_msgs::Twist cmd_msg;
            if (success)
            {
                cmd_msg.linear.x = next_cmd[0];
                cmd_msg.angular.z = next_cmd[1];

                last_valid_move_time_ = ros::Time::now();
                visualizeTrajectories(all_trajs, best_traj);
            }
            else
            {
                // 规划失败
                if (checkStuck())
                {
                    rate.sleep();
                    continue;
                }
                cmd_msg.linear.x = 0.0;
                cmd_msg.angular.z = 0.0;
            }

            cmd_pub_.publish(cmd_msg);
            rate.sleep();
        }
    }

    void DWAROSNode::visualizeTarget(const Eigen::Vector2d &target)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world"; // 假设在world坐标系下控制
        marker.header.stamp = ros::Time::now();
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = target.x();
        marker.pose.position.y = target.y();
        marker.pose.position.z = robot_z_;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.pose.orientation.w = 1.0;
        target_pub_.publish(marker);
    }

    void DWAROSNode::visualizeTrajectories(const std::vector<Trajectory> &all_trajs, const Trajectory &best_traj)
    {
        // 1. 发布最佳路径
        nav_msgs::Path path_msg;
        path_msg.header.stamp = ros::Time::now();
        path_msg.header.frame_id = "world";

        for (const auto &s : best_traj)
        {
            geometry_msgs::PoseStamped p;
            p.pose.position.x = s.x;
            p.pose.position.y = s.y;
            p.pose.position.z = robot_z_;
            p.pose.orientation.w = 1.0;
            path_msg.poses.push_back(p);
        }
        best_traj_pub_.publish(path_msg);

        // 2. 发布候选路径 (使用 LINE_LIST 降采样发布)
        visualization_msgs::MarkerArray markers;
        visualization_msgs::Marker line_marker;
        line_marker.header.frame_id = "world";
        line_marker.header.stamp = ros::Time::now();
        line_marker.ns = "candidates";
        line_marker.id = 0;
        line_marker.type = visualization_msgs::Marker::LINE_LIST;
        line_marker.scale.x = 0.02;
        line_marker.color.a = 0.2;
        line_marker.color.g = 1.0;
        line_marker.pose.orientation.w = 1.0;

        int count = 0;
        for (const auto &traj : all_trajs)
        {
            count++;
            if (count % 10 != 0)
                continue; // 降采样

            for (size_t i = 0; i < traj.size() - 1; ++i)
            {
                geometry_msgs::Point p1, p2;
                p1.x = traj[i].x;
                p1.y = traj[i].y;
                p1.z = robot_z_;
                p2.x = traj[i + 1].x;
                p2.y = traj[i + 1].y;
                p2.z = robot_z_;
                line_marker.points.push_back(p1);
                line_marker.points.push_back(p2);
            }
        }
        markers.markers.push_back(line_marker);
        traj_pub_.publish(markers);
    }

} // namespace custom_dwa_planner