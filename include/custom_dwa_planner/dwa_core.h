#ifndef CUSTOM_DWA_PLANNER_DWA_CORE_H
#define CUSTOM_DWA_PLANNER_DWA_CORE_H

#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <Eigen/Dense>
#include <fstream> // [新增] 引入文件流
#include <string>  // [新增]
#include "custom_dwa_planner/dwa_config.h"

namespace custom_dwa_planner
{

    // 机器人状态定义
    struct State
    {
        double x = 0.0;
        double y = 0.0;
        double yaw = 0.0;
        double v = 0.0; // 线速度
        double w = 0.0; // 角速度
    };

    // 轨迹定义 (用于可视化和计算)
    using Trajectory = std::vector<State>;

    // 轻量级地图数据封装 (避免拷贝)
    struct MapGrid
    {
        const int8_t *data = nullptr; // 指向 costmap 数据的指针
        int width = 0;
        int height = 0;
        double resolution = 0.05;
        double origin_x = 0.0;
        double origin_y = 0.0;
    };

    class DWACore
    {
    public:
        DWACore();
        ~DWACore(); // [修改] 需要析构函数来关闭文件

        /**
         * @brief 初始化配置
         */
        void updateConfig(const DWAConfig &config);

        /**
         * @brief 更新地图数据
         * 注意：调用者必须保证 data 指针在 plan 期间有效
         */
        void setMap(const int8_t *data, int width, int height, double res, double ox, double oy);

        /**
         * @brief [新增] 初始化调试日志
         * @param path 日志文件保存路径
         */
        void initDebugLog(const std::string &path);

        /**
         * @brief DWA 规划主函数
         * @param current_state 机器人当前状态 (Odom Frame)
         * @param local_goal    局部目标点 [x, y] (Odom Frame)
         * @param next_cmd      [输出] 计算出的最佳速度指令 [v, w]
         * @param best_traj     [输出] 最佳轨迹 (用于可视化)
         * @param all_trajs     [输出] 所有候选轨迹 (用于调试/可视化)
         * @return true 如果成功找到有效轨迹, false 如果陷入死路
         */
        bool computeVelocityCommands(const State &current_state,
                                     const Eigen::Vector2d &local_goal,
                                     Eigen::Vector2d &next_cmd,
                                     Trajectory &best_traj,
                                     std::vector<Trajectory> &all_trajs);

    private:
        // --- 内部辅助函数 ---

        /**
         * @brief 计算动态窗口 (Dynamic Window)
         * 基于当前速度和加速度限制，计算下一步可达的速度范围 [min_v, max_v, min_w, max_w]
         */
        std::vector<double> calcDynamicWindow(const State &state);

        /**
         * @brief 推演单条轨迹
         */
        Trajectory predictTrajectory(const State &init_state, double v, double w);

        /**
         * @brief 计算轨迹代价
         * @return 代价值 (float::max 表示不可行)
         */
        double calcTrajectoryCost(const Trajectory &traj, const Eigen::Vector2d &goal);

        /**
         * @brief 计算障碍物代价 (包含 Footprint 检查)
         */
        double calcObstacleCost(const Trajectory &traj);

        /**
         * @brief 检查单个点是否安全 (Grid Map 检查)
         * Python逻辑: -1(未知)视作安全, >阈值视作障碍
         */
        bool checkPointSafe(double wx, double wy);

        /**
         * @brief 获取栅格代价值
         */
        int8_t getGridValue(double wx, double wy);

        /**
         * @brief 角度归一化 [-PI, PI]
         */
        double normalizeAngle(double angle);

    private:
        DWAConfig config_;
        MapGrid map_;
        bool map_received_ = false;

        // [新增] 调试日志文件流
        std::ofstream debug_log_file_;
        bool log_initialized_ = false;
    };

} // namespace custom_dwa_planner

#endif // CUSTOM_DWA_PLANNER_DWA_CORE_H