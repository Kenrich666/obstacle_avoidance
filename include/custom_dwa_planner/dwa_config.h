#ifndef CUSTOM_DWA_PLANNER_DWA_CONFIG_H
#define CUSTOM_DWA_PLANNER_DWA_CONFIG_H

#include <vector>
#include <Eigen/Dense>

namespace custom_dwa_planner {

/**
 * @brief DWA 算法配置参数结构体
 * 对应 Python 脚本中的 DWAConfig 类
 */
struct DWAConfig {
    // --- 物理限制 (Physical Limits) ---
    double max_vel_x = 1.0;       // [m/s] 最大前向速度
    double min_vel_x = -0.2;      // [m/s] 最小前向速度（允许微量倒车）
    double max_rot_vel = 0.6;     // [rad/s] 最大旋转速度
    
    double acc_lim_x = 10.0;      // [m/s^2] X轴加速度限制
    double acc_lim_theta = 3.5;   // [rad/s^2] 旋转加速度限制

    // --- DWA 采样参数 (Sampling) ---
    double sim_time = 2.5;        // [s] 轨迹预测时间
    double sim_dt = 0.1;          // [s] 积分步长
    
    // 采样分辨率 (Python: v_res, w_res)
    // 在 C++ 实现中，为了精度控制，我们通常将其转换为采样步数或保留分辨率逻辑
    double v_res = 0.05;          // [m/s] 线速度分辨率
    double w_res = 0.05;          // [rad/s] 角速度分辨率

    // --- 评分权重 (Scoring) ---
    double path_distance_bias = 32.0;  // 贴近路径权重 (heading)
    double goal_distance_bias = 20.0;  // 趋向目标权重 (dist)
    double occdist_scale = 0.04;       // 避障权重

    // --- 路径跟随 ---
    double lookahead_dist = 4.0;       // [m] 胡萝卜距离
    double goal_reach_dist = 1;      // [m] 到达判定距离
    double stuck_timeout = 3.0;        // [s] 堵塞超时时间

    // --- 机器人足迹 (Footprint) ---
    // 使用 Eigen::Vector2d 存储多边形点，相对于 Base Link
    std::vector<Eigen::Vector2d> footprint;

    // --- 代价地图参数 ---
    int costmap_obstacle_threshold = 80; // >80 或 -1(未知) 的逻辑将在Core中特殊处理

    DWAConfig() {
        // 初始化默认矩形足迹 (对应 Python 中的 list)
        footprint = {
            {0.5, 0.4}, {0.5, -0.4}, {-0.5, 0.4}, {-0.5, -0.4},
            {0.5, 0.0}, {-0.5, 0.0}, {0.0, 0.4}, {0.0, -0.4}, {0.0, 0.0}
        };
    }
};

} // namespace custom_dwa_planner

#endif // CUSTOM_DWA_PLANNER_DWA_CONFIG_H