#include "custom_dwa_planner/dwa_core.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip> // [新增] 用于格式化输出

namespace custom_dwa_planner
{

    DWACore::DWACore()
    {
        // 构造函数可以留空，或者做一些基础初始化
    }

    // [新增] 析构函数关闭文件
    DWACore::~DWACore()
    {
        if (debug_log_file_.is_open())
        {
            debug_log_file_.close();
        }
    }

    // [新增] 初始化日志
    void DWACore::initDebugLog(const std::string &path)
    {
        // std::ios::trunc 确保每次启动都清空文件
        debug_log_file_.open(path, std::ios::out | std::ios::trunc);
        if (debug_log_file_.is_open())
        {
            // 写入 CSV 表头
            debug_log_file_ << "Selected_V,Selected_W,"
                            << "Raw_Goal_Dist,Weighted_Goal_Cost,"
                            << "Raw_Speed_Diff,Weighted_Speed_Cost,"
                            << "Raw_Obs_Cost,Weighted_Obs_Cost,"
                            << "Total_Cost" << std::endl;
            log_initialized_ = true;
        }
        else
        {
            std::cerr << "[DWACore] Failed to open debug log file: " << path << std::endl;
        }
    }

    void DWACore::updateConfig(const DWAConfig &config)
    {
        config_ = config;
    }

    void DWACore::setMap(const int8_t *data, int width, int height, double res, double ox, double oy)
    {
        map_.data = data;
        map_.width = width;
        map_.height = height;
        map_.resolution = res;
        map_.origin_x = ox;
        map_.origin_y = oy;
        map_received_ = true;
    }

    double DWACore::normalizeAngle(double angle)
    {
        while (angle > M_PI)
            angle -= 2.0 * M_PI;
        while (angle < -M_PI)
            angle += 2.0 * M_PI;
        return angle;
    }

    bool DWACore::checkPointSafe(double wx, double wy)
    {
        if (!map_received_ || map_.data == nullptr)
            return false;

        // 使用 floor 确保负数坐标也能正确映射
        int mx = static_cast<int>(std::floor((wx - map_.origin_x) / map_.resolution));
        int my = static_cast<int>(std::floor((wy - map_.origin_y) / map_.resolution));

        if (mx >= 0 && mx < map_.width && my >= 0 && my < map_.height)
        {
            int index = my * map_.width + mx;
            int8_t val = map_.data[index];

            // Python 逻辑复刻: -1 (未知) 视为安全
            if (val == -1)
            {
                return true;
            }
            // 大于阈值视为障碍
            if (val > config_.costmap_obstacle_threshold)
            {
                return false;
            }
            return true;
        }

        // 出界视为不安全 (或者可以根据需求改为安全，这里保持保守策略)
        return false;
    }

    int8_t DWACore::getGridValue(double wx, double wy)
    {
        if (!map_received_ || map_.data == nullptr)
            return 100;

        int mx = static_cast<int>(std::floor((wx - map_.origin_x) / map_.resolution));
        int my = static_cast<int>(std::floor((wy - map_.origin_y) / map_.resolution));

        if (mx >= 0 && mx < map_.width && my >= 0 && my < map_.height)
        {
            int index = my * map_.width + mx;
            int8_t val = map_.data[index];
            // 如果是 -1 (未知)，代价设为 0 (Python逻辑)
            return (val == -1) ? 0 : val;
        }
        return 100; // 出界代价最大
    }

    std::vector<double> DWACore::calcDynamicWindow(const State &state)
    {
        // 1. 物理速度限制
        double vs_min_v = config_.min_vel_x;
        double vs_max_v = config_.max_vel_x;
        double vs_min_w = -config_.max_rot_vel;
        double vs_max_w = config_.max_rot_vel;

        // 2. 加速度限制下的可达范围
        double vd_min_v = state.v - config_.acc_lim_x * config_.sim_dt;
        double vd_max_v = state.v + config_.acc_lim_x * config_.sim_dt;
        double vd_min_w = state.w - config_.acc_lim_theta * config_.sim_dt;
        double vd_max_w = state.w + config_.acc_lim_theta * config_.sim_dt;

        // 3. 取交集
        std::vector<double> dw(4);
        dw[0] = std::max(vs_min_v, vd_min_v); // min_v
        dw[1] = std::min(vs_max_v, vd_max_v); // max_v
        dw[2] = std::max(vs_min_w, vd_min_w); // min_w
        dw[3] = std::min(vs_max_w, vd_max_w); // max_w

        return dw;
    }

    Trajectory DWACore::predictTrajectory(const State &init_state, double v, double w)
    {
        Trajectory traj;
        // 预分配内存以优化性能
        int steps = static_cast<int>(config_.sim_time / config_.sim_dt);
        traj.reserve(steps + 1);

        State state = init_state;
        double time = 0.0;

        // 初始状态不一定需要加入，取决于是否需要从当前点开始画线
        // 这里加入初始点以便碰撞检测更严谨
        // traj.push_back(state);

        while (time <= config_.sim_time)
        {
            // 运动学模型
            state.x += v * std::cos(state.yaw) * config_.sim_dt;
            state.y += v * std::sin(state.yaw) * config_.sim_dt;
            state.yaw += w * config_.sim_dt;
            state.yaw = normalizeAngle(state.yaw);

            state.v = v;
            state.w = w;

            traj.push_back(state);
            time += config_.sim_dt;
        }
        return traj;
    }

    double DWACore::calcObstacleCost(const Trajectory &traj)
    {
        double total_cost = 0.0;
        int skip = 2;

        for (size_t i = 0; i < traj.size(); i += skip)
        {
            const auto &p = traj[i];

            // --- 改进版逻辑 ---
            double current_step_max_cost = 0.0;

            // 1. 检查中心点
            int8_t center_val = getGridValue(p.x, p.y);
            if (center_val > config_.costmap_obstacle_threshold)
                return std::numeric_limits<double>::infinity();
            current_step_max_cost = std::max(current_step_max_cost, (double)center_val);

            // 2. 检查 Footprint 并寻找最大代价
            double cos_yaw = std::cos(p.yaw);
            double sin_yaw = std::sin(p.yaw);

            for (const auto &pt : config_.footprint)
            {
                double global_x = (pt.x() * cos_yaw - pt.y() * sin_yaw) + p.x;
                double global_y = (pt.x() * sin_yaw + pt.y() * cos_yaw) + p.y;

                int8_t fp_val = getGridValue(global_x, global_y);

                // 硬阈值检查 (Dead check)
                if (fp_val > config_.costmap_obstacle_threshold)
                {
                    return std::numeric_limits<double>::infinity();
                }

                // 软代价记录 (Soft penalty)：记录这一步中 footprint 碰到的最大代价值
                current_step_max_cost = std::max(current_step_max_cost, (double)fp_val);
            }

            // 将这一步的最大潜在危险加入总代价
            total_cost += current_step_max_cost;
        }
        return total_cost;
    }

    bool DWACore::computeVelocityCommands(const State &current_state,
                                          const Eigen::Vector2d &local_goal,
                                          Eigen::Vector2d &next_cmd,
                                          Trajectory &best_traj,
                                          std::vector<Trajectory> &all_trajs)
    {

        if (!map_received_)
            return false;

        best_traj.clear();
        all_trajs.clear();

        std::vector<double> dw = calcDynamicWindow(current_state);

        double min_cost = std::numeric_limits<double>::infinity();
        bool found_valid_traj = false;

        // 用于记录最佳路径的详细评分组件
        struct ScoreComponents
        {
            double raw_goal = 0.0;
            double weighted_goal = 0.0;
            double raw_speed = 0.0;
            double weighted_speed = 0.0;
            double raw_obs = 0.0;
            double weighted_obs = 0.0;
        } best_scores;

        // ... (采样步骤计算逻辑保持不变) ...
        int steps_v = std::max(2, static_cast<int>((dw[1] - dw[0]) / config_.v_res) + 1);
        int steps_w = std::max(2, static_cast<int>((dw[3] - dw[2]) / config_.w_res) + 1);

        std::vector<double> v_samples, w_samples;
        // ... (v_samples 和 w_samples 生成逻辑保持不变) ...
        if (steps_v > 1)
        {
            double step = (dw[1] - dw[0]) / (steps_v - 1);
            for (int i = 0; i < steps_v; ++i)
                v_samples.push_back(dw[0] + i * step);
        }
        else
        {
            v_samples.push_back(dw[0]);
        }

        if (steps_w > 1)
        {
            double step = (dw[3] - dw[2]) / (steps_w - 1);
            for (int i = 0; i < steps_w; ++i)
                w_samples.push_back(dw[2] + i * step);
        }
        else
        {
            w_samples.push_back(dw[2]);
        }

        for (double v : v_samples)
        {
            for (double w : w_samples)
            {
                Trajectory traj = predictTrajectory(current_state, v, w);
                if (traj.empty())
                    continue;

                const auto &last_pose = traj.back();

                // --- 评分逻辑 ---
                double dx = local_goal.x() - last_pose.x;
                double dy = local_goal.y() - last_pose.y;
                double raw_goal_dist = std::hypot(dx, dy); // 原始距离

                double raw_speed_diff = config_.max_vel_x - last_pose.v; // 原始速度差

                double raw_ob_cost = calcObstacleCost(traj); // 原始障碍物代价

                if (raw_ob_cost == std::numeric_limits<double>::infinity())
                {
                    continue;
                }

                // 加权分项
                double weighted_goal = config_.goal_distance_bias * raw_goal_dist;
                double weighted_speed = config_.path_distance_bias * raw_speed_diff;
                double weighted_obs = config_.occdist_scale * raw_ob_cost;

                double final_cost = weighted_goal + weighted_speed + weighted_obs;

                all_trajs.push_back(traj);

                if (final_cost < min_cost)
                {
                    min_cost = final_cost;
                    next_cmd[0] = v;
                    next_cmd[1] = w;
                    best_traj = traj;
                    found_valid_traj = true;

                    // [新增] 暂存最佳评分详情
                    best_scores.raw_goal = raw_goal_dist;
                    best_scores.weighted_goal = weighted_goal;
                    best_scores.raw_speed = raw_speed_diff;
                    best_scores.weighted_speed = weighted_speed;
                    best_scores.raw_obs = raw_ob_cost;
                    best_scores.weighted_obs = weighted_obs;
                }
            }
        }

        // [新增] 写入日志
        if (log_initialized_ && found_valid_traj)
        {
            debug_log_file_ << std::fixed << std::setprecision(4)
                            << next_cmd[0] << "," << next_cmd[1] << ","
                            << best_scores.raw_goal << "," << best_scores.weighted_goal << ","
                            << best_scores.raw_speed << "," << best_scores.weighted_speed << ","
                            << best_scores.raw_obs << "," << best_scores.weighted_obs << ","
                            << min_cost << std::endl;
        }

        return found_valid_traj;
    }

} // namespace custom_dwa_planner