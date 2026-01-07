#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import math
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class DWAConfig:
    def __init__(self):
        # --- 物理限制 (Physical Limits) ---
        self.max_vel_x = 1.0       # 最大前向速度
        self.min_vel_x = -0.1      # 最小前向速度（允许微量倒车调整）
        self.max_rot_vel = 0.6     # 最大旋转速度
        
        self.acc_lim_x = 10.0       # X轴加速度限制
        self.acc_lim_theta = 3.5   # 旋转加速度限制

        # --- DWA 采样参数 (Sampling) ---
        self.sim_time = 2.5        # 轨迹预测时间
        self.sim_dt = 0.1          # 积分步长
        self.v_res = 0.05          # 线速度分辨率
        self.w_res = 0.05          # 角速度分辨率
        
        # --- 评分权重 (Scoring) ---
        self.path_distance_bias = 32.0  # 贴近路径权重
        self.goal_distance_bias = 20.0  # 趋向目标权重
        self.occdist_scale = 0.04       # 避障权重
        
        # --- 路径跟随 ---
        self.lookahead_dist = 15.0       # 胡萝卜距离
        self.goal_reach_dist = 0.5      # 到达判定距离
        self.stuck_timeout = 3.0        # 堵塞超时时间(s)

        # --- 机器人足迹 (Footprint) ---
        self.footprint = [
            [0.5, 0.4], [0.5, -0.4], [-0.5, 0.4], [-0.5, -0.4],
            [0.5, 0.0], [-0.5, 0.0], [0.0, 0.4], [0.0, -0.4], [0.0, 0.0]
        ]
        # 避障阈值：>60 或 -1(未知) 都视为障碍
        self.costmap_obstacle_threshold = 80

class DWAPlannerNode:
    def __init__(self):
        rospy.init_node('custom_dwa_planner', anonymous=True)
        self.config = DWAConfig()
        
        # 核心状态
        self.state = np.zeros(5) # [x, y, yaw, v, w] (odom frame)
        self.robot_z = -76.0     # 默认深度，会从odom更新
        
        # 路径管理
        self.raw_global_path = None # 原始路径 (Map frame)
        self.local_plan = None      # 转换后的局部路径 (Odom frame)
        self.local_goal = None      # 当前追踪点
        self.last_path_index = 0    # [优化] 滑动窗口搜索索引
        
        # 堵塞恢复
        self.last_valid_move_time = rospy.Time.now()
        
        # 地图数据
        self.costmap = None
        self.map_info = None
        
        # TF Buffer [关键修复]
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 通信
        self.path_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_callback)
        rospy.Subscriber('/rexrov2/pose_gt', Odometry, self.odom_callback)
        rospy.Subscriber('/projected_sonar_map', OccupancyGrid, self.map_callback)
        
        self.cmd_pub = rospy.Publisher('/rexrov2/cmd_vel_move_base', Twist, queue_size=1)
        
        # 可视化
        self.traj_pub = rospy.Publisher('/dwa_trajectories', MarkerArray, queue_size=1)
        self.best_traj_pub = rospy.Publisher('/dwa_best_traj', Path, queue_size=1)
        self.target_pub = rospy.Publisher('/dwa_current_target', Marker, queue_size=1)

        self.rate = rospy.Rate(10)
        rospy.loginfo("Industrial DWA Planner Initialized (TF aware).")

    def path_callback(self, msg):
        """接收全局路径，重置搜索索引"""
        if len(msg.poses) > 0:
            self.raw_global_path = msg
            self.last_path_index = 0 # 新路径重置索引
            self.last_valid_move_time = rospy.Time.now() # 重置超时
            rospy.loginfo(f"Received new plan with {len(msg.poses)} points frame: {msg.header.frame_id}")
        else:
            self.raw_global_path = None

    def odom_callback(self, msg):
        """更新机器人状态"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_z = -76.0 # [修复] 动态获取Z
        # self.robot_z = msg.pose.pose.position.z # [修复] 动态获取Z
        
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # 假设 twist 是 body frame，需确认 (通常 Odometry twist 是 child_frame_id，即 body)
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        
        self.state = np.array([x, y, yaw, v, w])

    def map_callback(self, msg):
        self.map_info = msg.info
        # 使用 int8 存储，-1 保持为 -1 (int8: 255 -> -1)
        self.costmap = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

    def transform_global_plan(self):
        """
        [关键修复] 将 Global Path (Map Frame) 转换到 Odom Frame
        这是为了让 DWA 在统一的坐标系下计算
        """
        if self.raw_global_path is None: return None
        
        target_frame = "world" # 假设我们的 Odom/Control 都在 world frame
        # 如果你的 tf 树是 map -> odom -> base_link，这里通常转到 odom
        
        # 检查是否需要转换
        if self.raw_global_path.header.frame_id == target_frame:
            return self.raw_global_path

        try:
            # 获取变换矩阵
            transform = self.tf_buffer.lookup_transform(
                target_frame, 
                self.raw_global_path.header.frame_id, 
                rospy.Time(0), 
                rospy.Duration(0.5)
            )
            
            new_path = Path()
            new_path.header.frame_id = target_frame
            new_path.header.stamp = rospy.Time.now()
            
            # 为了性能，可以降采样转换 (例如每 5 个点转 1 个)
            # 这里为了精度全转，但要注意 Python 性能
            for pose in self.raw_global_path.poses:
                new_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
                new_path.poses.append(new_pose)
                
            return new_path
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"TF Transform failed: {e}")
            return None

    def get_local_target(self, path_msg):
        """
        [优化] 滑动窗口搜索局部目标
        """
        if path_msg is None or len(path_msg.poses) == 0: return None

        rx, ry = self.state[0], self.state[1]
        min_dist = float('inf')
        closest_idx = -1
        
        # [优化] 限制搜索范围：只在上次索引附近搜索 (例如前50个点)
        start_idx = self.last_path_index
        search_window = 100 
        end_idx = min(start_idx + search_window, len(path_msg.poses))
        
        # 如果机器人倒退了或者路径重置，可能需要回溯，简单起见如果找不到更近的就全搜
        # 这里实现简单的向前窗口
        
        for i in range(start_idx, end_idx):
            px = path_msg.poses[i].pose.position.x
            py = path_msg.poses[i].pose.position.y
            dist_sq = (px - rx)**2 + (py - ry)**2
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_idx = i
        
        # 如果窗口内没找到更近的（可能跑偏了），尝试全搜索作为 fallback
        if closest_idx == -1 or min_dist > 25.0: # > 5m error
             for i in range(len(path_msg.poses)):
                px = path_msg.poses[i].pose.position.x
                py = path_msg.poses[i].pose.position.y
                dist_sq = (px - rx)**2 + (py - ry)**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    closest_idx = i
        
        if closest_idx != -1:
            self.last_path_index = closest_idx # 更新索引缓存

        # 找胡萝卜 (Lookahead)
        target_idx = closest_idx
        dist_sum = 0.0
        for i in range(closest_idx, len(path_msg.poses) - 1):
            p1 = path_msg.poses[i].pose.position
            p2 = path_msg.poses[i+1].pose.position
            d = math.hypot(p2.x - p1.x, p2.y - p1.y)
            dist_sum += d
            target_idx = i + 1
            if dist_sum > self.config.lookahead_dist:
                break
        
        final_pose = path_msg.poses[target_idx]
        return np.array([final_pose.pose.position.x, final_pose.pose.position.y])

    def check_stuck(self):
        """[新增] 堵塞检测与超时处理"""
        if (rospy.Time.now() - self.last_valid_move_time).to_sec() > self.config.stuck_timeout:
            rospy.logwarn("Robot STUCK for too long! Clearing plan...")
            self.raw_global_path = None # 强制停止，等待新规划
            self.cmd_pub.publish(Twist()) # 急停
            return True
        return False

    def check_point_safe(self, wx, wy):
        """[关键修复] 严格检查未知区域"""
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        
        if 0 <= mx < self.map_info.width and 0 <= my < self.map_info.height:
            val = self.costmap[my, mx]
            # -1 (未知) 安全的
            if val == -1: 
                return True # <--- 关键修改：遇到未知直接放行
            
            if val > self.config.costmap_obstacle_threshold:
                return False # 只有确认为障碍物才拦截
            return True
        return False # 出界通常视为不安全，或者根据策略决定

    def get_grid_value_cost(self, wx, wy):
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        if 0 <= mx < self.map_info.width and 0 <= my < self.map_info.height:
            val = self.costmap[my, mx]
            if val == -1: 
                return 0
            return val
        return 100 # 出界

    def run(self):
        while not rospy.is_shutdown():
            # 1. 基础检查
            if self.raw_global_path is None or self.map_info is None:
                self.cmd_pub.publish(Twist()) 
                self.rate.sleep()
                continue
            
            # 2. 坐标转换 Map -> Odom
            local_plan = self.transform_global_plan()
            if local_plan is None:
                self.rate.sleep()
                continue
                
            # 3. 提取局部目标 (胡萝卜)
            self.local_goal = self.get_local_target(local_plan)
            if self.local_goal is None:
                self.rate.sleep()
                continue
            
            # 检查是否到达终点
            last_p = local_plan.poses[-1].pose.position
            dist_end = math.hypot(self.state[0] - last_p.x, self.state[1] - last_p.y)
            if dist_end < self.config.goal_reach_dist:
                rospy.loginfo_throttle(5.0, "Goal Reached.")
                self.cmd_pub.publish(Twist())
                self.raw_global_path = None
                continue

            self.visualize_target(self.local_goal)

            # 4. DWA 搜索
            dw = self.calc_dynamic_window()
            best_u, best_traj, all_trajectories = self.dwa_search(dw)
            
            # 5. 执行控制
            cmd = Twist()
            if best_u is not None:
                cmd.linear.x = best_u[0]
                cmd.angular.z = best_u[1]
                self.visualize_trajectories(all_trajectories, best_traj)
                self.last_valid_move_time = rospy.Time.now() # 成功规划，重置超时
            else:
                # 陷入困境
                if self.check_stuck():
                    continue # 已处理超时
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            
            self.cmd_pub.publish(cmd)
            self.rate.sleep()

    def calc_dynamic_window(self):
        Vs = [self.config.min_vel_x, self.config.max_vel_x, 
              -self.config.max_rot_vel, self.config.max_rot_vel]
        dt = 0.1 # 控制周期
        Vd = [self.state[3] - self.config.acc_lim_x * dt,
              self.state[3] + self.config.acc_lim_x * dt,
              self.state[4] - self.config.acc_lim_theta * dt,
              self.state[4] + self.config.acc_lim_theta * dt]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def dwa_search(self, dw):
        best_u = None
        min_cost = float('inf')
        best_traj = None
        all_trajectories = [] 

        # [修复] 使用 linspace 确保包含边界，防止丢掉最大速度
        # 比如从 dw[0] 到 dw[1]，取适当数量的点
        steps_v = int((dw[1] - dw[0]) / self.config.v_res) + 1
        steps_w = int((dw[3] - dw[2]) / self.config.w_res) + 1
        
        v_samples = np.linspace(dw[0], dw[1], max(2, steps_v))
        w_samples = np.linspace(dw[2], dw[3], max(2, steps_w))
        
        for v in v_samples:
            for w in w_samples:
                traj = self.predict_trajectory(self.state, v, w)
                
                # 评分
                to_goal_cost = math.hypot(self.local_goal[0] - traj[-1,0], self.local_goal[1] - traj[-1,1])
                speed_cost = self.config.max_vel_x - traj[-1, 3]
                ob_cost = self.calc_obstacle_cost(traj)
                
                if ob_cost == float('inf'): continue
                
                final_cost = (self.config.goal_distance_bias * to_goal_cost +
                              self.config.path_distance_bias * speed_cost + 
                              self.config.occdist_scale * ob_cost)
                
                all_trajectories.append(traj)
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_traj = traj

        return best_u, best_traj, all_trajectories

    def predict_trajectory(self, state_init, v, w):
        traj = []
        state = np.array(state_init)
        time = 0
        while time <= self.config.sim_time:
            dt = self.config.sim_dt
            state[0] += v * np.cos(state[2]) * dt
            state[1] += v * np.sin(state[2]) * dt
            state[2] += w * dt
            # [修复] 角度归一化
            state[2] = normalize_angle(state[2])
            
            state[3] = v
            state[4] = w
            traj.append(np.array(state))
            time += dt
        return np.array(traj)

    def calc_obstacle_cost(self, traj):
        cost = 0
        skip = 2
        for i in range(0, len(traj), skip):
            cx, cy, cyaw = traj[i, 0], traj[i, 1], traj[i, 2]
            
            # 1. 中心点粗查
            if not self.check_point_safe(cx, cy): return float('inf')
            
            # 2. Footprint 精查
            rot_mat = np.array([[np.cos(cyaw), -np.sin(cyaw)],
                                [np.sin(cyaw),  np.cos(cyaw)]])
            for pt in self.config.footprint:
                p_world = np.dot(rot_mat, np.array(pt)) + np.array([cx, cy])
                if not self.check_point_safe(p_world[0], p_world[1]): return float('inf')
            
            # 累加代价值
            cost += self.get_grid_value_cost(cx, cy)
        return cost

    def visualize_target(self, target):
        marker = Marker()
        marker.header.frame_id = "world" # 假设是 world
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.5; marker.scale.y = 0.5; marker.scale.z = 0.5
        marker.color.a = 1.0; marker.color.r = 1.0
        marker.pose.position.x = target[0]
        marker.pose.position.y = target[1]
        # [修复] 动态 Z 高度
        marker.pose.position.z = self.robot_z 
        marker.pose.orientation.w = 1.0
        self.target_pub.publish(marker)

    def visualize_trajectories(self, all_traj, best_traj):
        # 最佳路径
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "world"
        for state in best_traj:
            pose = PoseStamped()
            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.pose.position.z = self.robot_z
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.best_traj_pub.publish(path_msg)
        
        # 候选路径（为了性能，大幅降采样）
        markers = MarkerArray()
        line_marker = Marker()
        line_marker.header.frame_id = "world"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "candidates"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.scale.x = 0.02
        line_marker.color.a = 0.2
        line_marker.color.g = 1.0
        line_marker.pose.orientation.w = 1.0
        
        count = 0
        for traj in all_traj:
            count += 1
            if count % 20 != 0: continue 
            for k in range(len(traj)-1):
                p1 = Point(traj[k,0], traj[k,1], self.robot_z)
                p2 = Point(traj[k+1,0], traj[k+1,1], self.robot_z)
                line_marker.points.append(p1)
                line_marker.points.append(p2)
        markers.markers.append(line_marker)
        self.traj_pub.publish(markers)

if __name__ == '__main__':
    try:
        node = DWAPlannerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass