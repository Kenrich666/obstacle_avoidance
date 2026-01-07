#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
import tf2_ros
import tf2_py as tf2
from nav_msgs.msg import OccupancyGrid, Odometry
from marine_acoustic_msgs.msg import ProjectedSonarImage
from std_msgs.msg import Header

class DirectSonarMapper:
    def __init__(self):
        rospy.init_node('direct_sonar_mapper', anonymous=True)
        
        # --- 参数配置 ---
        self.sonar_topic = '/rexrov2/blueview_p900/sonar_image_raw'
        self.odom_topic = '/rexrov2/pose_gt'
        self.map_topic = '/projected_sonar_map'
        
        self.map_frame = "world"
        self.robot_base_frame = "rexrov2/base_link"
        self.override_sonar_frame = "blueview_p900_link" 

        # 地图参数
        self.map_res = 0.2
        self.map_width_m = 300.0
        self.map_height_m = 300.0
        self.map_origin_x = -150.0
        self.map_origin_y = -150.0
        self.map_z_level = 0.0
        
        # 声呐过滤参数
        self.min_range = 1.0        
        self.intensity_threshold = 10 
        
        # 映射逻辑参数
        self.sonar_max_range = 16.0 
        self.sonar_fov_rad = np.deg2rad(95.0) 
        self.fov_deg = 90.0
        
        self.hit_increment = 30.0
        self.max_occupancy = 100.0
        
        # [优化1] 基础衰减率调高，让旧障碍物消失得更快
        self.base_decay_rate = 5.0  
        
        self.velocity_decay_factor = 1.0
        
        # [优化2] 新增：旋转衰减因子
        # 旋转时，显著增加清除速度，利用“扫视”动作快速擦除错误障碍物
        self.angular_decay_factor = 5.0 
        
        # [优化3] 新增：最大建图旋转速度 (rad/s)
        # 超过约 10度/秒 (0.17 rad/s) 时，声纳数据畸变严重，停止写入障碍物
        self.max_mapping_yaw_rate = 0.15 

        self.dilation_radius = 1
        self.max_tilt_deg = 5.0

        # --- 内部变量 ---
        self.grid_shape = (int(self.map_height_m / self.map_res), 
                           int(self.map_width_m / self.map_res))
        self.occupancy_grid = np.zeros(self.grid_shape, dtype=np.float32)
        
        self.current_linear_speed = 0.0
        self.current_yaw_rate = 0.0 # 新增：当前旋转速度
        
        self.last_update_time = None
        self.cached_angles = None
        self.last_beam_count = 0

        self._init_precomputed_tables()

        # --- TF & ROS ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.Subscriber(self.sonar_topic, ProjectedSonarImage, self.sonar_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        
        self.map_pub = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1)
        
        rospy.loginfo("抗拖影版声呐建图节点已启动.")

    def _init_precomputed_tables(self):
        self.range_indices = int(self.sonar_max_range / self.map_res) + 1
        x_range = np.arange(-self.range_indices, self.range_indices + 1)
        y_range = np.arange(-self.range_indices, self.range_indices + 1)
        self.cache_grid_x, self.cache_grid_y = np.meshgrid(x_range, y_range)
        
        rel_world_x = self.cache_grid_x * self.map_res
        rel_world_y = self.cache_grid_y * self.map_res
        
        self.cache_dist_sq = rel_world_x**2 + rel_world_y**2
        self.cache_angles = np.arctan2(rel_world_y, rel_world_x)

    def odom_callback(self, msg):
        # 提取线速度
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_linear_speed = np.sqrt(vx**2 + vy**2)
        
        # [关键] 提取角速度 (Yaw Rate)
        self.current_yaw_rate = abs(msg.twist.twist.angular.z)

    def sonar_callback(self, data):
        current_time = rospy.Time.now()
        if self.last_update_time is None:
            self.last_update_time = current_time
            return 

        dt = (current_time - self.last_update_time).to_sec()
        if dt < 0: dt = 0.0
        self.last_update_time = current_time

        try:
            # 姿态检查
            base_trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_base_frame, data.header.stamp, rospy.Duration(0.5))
            
            bx, by, bz, bw = (base_trans.transform.rotation.x, base_trans.transform.rotation.y,
                              base_trans.transform.rotation.z, base_trans.transform.rotation.w)
            roll, pitch, _ = self.get_euler_from_quaternion(bx, by, bz, bw)
            
            if abs(roll) > np.deg2rad(self.max_tilt_deg) or abs(pitch) > np.deg2rad(self.max_tilt_deg):
                self.publish_map(data.header.stamp)
                return

            sensor_trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.override_sonar_frame, data.header.stamp, rospy.Duration(0.5))
            
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"TF Error: {e}")
            return

        tx = sensor_trans.transform.translation.x
        ty = sensor_trans.transform.translation.y
        tz = sensor_trans.transform.translation.z
        qx, qy, qz, qw = (sensor_trans.transform.rotation.x, sensor_trans.transform.rotation.y,
                          sensor_trans.transform.rotation.z, sensor_trans.transform.rotation.w)
        
        _, _, sensor_yaw = self.get_euler_from_quaternion(qx, qy, qz, qw)
        
        # --- 策略 A: 旋转过快时，仅执行衰减，不执行建图 ---
        # 旋转时TF误差和声呐物理畸变最大，不更新障碍物是消除拖影最有效的手段
        is_rotating_fast = self.current_yaw_rate > self.max_mapping_yaw_rate

        if not is_rotating_fast:
            # 只有转得慢的时候，才计算和添加障碍物
            self.process_obstacles(data, tx, ty, tz, qx, qy, qz, qw)
            # --- 策略 B: 衰减逻辑只在慢速旋转时执行 ---
            self.apply_fov_decay_optimized(dt, tx, ty, sensor_yaw)
        else:
            # 触发高速旋转保护时的日志
            # 使用 logwarn 以黄色高亮显示，throttle(1.0) 表示每秒最多打印一次
            rospy.logwarn_throttle(1.0, 
                f"[Anti-Blur] 旋转速度过高: {self.current_yaw_rate:.2f} > {self.max_mapping_yaw_rate:.2f} rad/s 已暂停建图。")
        
        self.publish_map(data.header.stamp)

    def process_obstacles(self, data, tx, ty, tz, qx, qy, qz, qw):
        rotation_matrix = self.quaternion_to_matrix(qx, qy, qz, qw)
        translation_vec = np.array([tx, ty, tz])

        ranges = np.array(data.ranges, dtype=np.float32)
        raw_image = np.frombuffer(data.image.data, dtype=np.uint8)
        n_ranges = len(ranges)
        
        if n_ranges == 0 or len(raw_image) % n_ranges != 0: return
        n_beams = int(len(raw_image) / n_ranges)
        
        try:
            matrix_data = raw_image.reshape((n_ranges, n_beams))
        except ValueError: return

        if self.cached_angles is None or n_beams != self.last_beam_count:
            fov_rad = np.deg2rad(self.fov_deg)
            self.cached_angles = np.linspace(fov_rad/2, -fov_rad/2, n_beams)
            self.last_beam_count = n_beams
        angles = self.cached_angles

        range_mask = ranges > self.min_range
        valid_matrix = matrix_data[range_mask, :]
        valid_ranges = ranges[range_mask]
        
        if valid_matrix.size == 0: return

        intensity_mask = valid_matrix > self.intensity_threshold
        
        if np.any(intensity_mask):
            R_grid, Theta_grid = np.meshgrid(valid_ranges, angles, indexing='ij')
            final_r = R_grid[intensity_mask]
            final_theta = Theta_grid[intensity_mask]
            
            X_sensor = final_r * np.cos(final_theta)
            Y_sensor = final_r * np.sin(final_theta)
            Z_sensor = np.zeros_like(X_sensor)
            
            points_sensor = np.column_stack((X_sensor, Y_sensor, Z_sensor))
            points_world = np.dot(points_sensor, rotation_matrix.T) + translation_vec
            
            j_indices = ((points_world[:, 0] - self.map_origin_x) / self.map_res).astype(int)
            i_indices = ((points_world[:, 1] - self.map_origin_y) / self.map_res).astype(int)
            
            valid_idx_mask = (i_indices >= 0) & (i_indices < self.grid_shape[0]) & \
                             (j_indices >= 0) & (j_indices < self.grid_shape[1])
            
            valid_i = i_indices[valid_idx_mask]
            valid_j = j_indices[valid_idx_mask]
            
            if len(valid_i) > 0:
                current_hits = np.zeros(self.grid_shape, dtype=bool)
                current_hits[valid_i, valid_j] = True
                dilated_hits = self.dilate_mask(current_hits, self.dilation_radius)
                
                self.occupancy_grid[dilated_hits] += self.hit_increment
                self.occupancy_grid[dilated_hits] = np.clip(self.occupancy_grid[dilated_hits], 0, self.max_occupancy)

    def apply_fov_decay_optimized(self, dt, sensor_x, sensor_y, sensor_yaw):
        if dt <= 0: return

        cx = int((sensor_x - self.map_origin_x) / self.map_res)
        cy = int((sensor_y - self.map_origin_y) / self.map_res)

        pad = self.range_indices
        min_map_x = max(0, cx - pad)
        max_map_x = min(self.grid_shape[1], cx + pad + 1)
        min_map_y = max(0, cy - pad)
        max_map_y = min(self.grid_shape[0], cy + pad + 1)
        
        if min_map_x >= max_map_x or min_map_y >= max_map_y: return

        # 计算对应的 Cache 索引
        cache_min_x = max(0, -(cx - pad))
        cache_max_x = cache_min_x + (max_map_x - min_map_x)
        cache_min_y = max(0, -(cy - pad))
        cache_max_y = cache_min_y + (max_map_y - min_map_y)

        roi_dist_sq = self.cache_dist_sq[cache_min_y:cache_max_y, cache_min_x:cache_max_x]
        roi_angles = self.cache_angles[cache_min_y:cache_max_y, cache_min_x:cache_max_x]

        delta_angle = roi_angles - sensor_yaw
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi

        # [策略 C] 在衰减时，视场角稍微放宽一点点 (FOV + 2度)
        # 这样可以在旋转时更积极地清除边缘的“幽灵点”
        decay_fov_rad = self.sonar_fov_rad + np.deg2rad(2.0)
        decay_max_range = self.sonar_max_range + 1.0
        
        fov_mask = (roi_dist_sq < decay_max_range**2) & \
                   (np.abs(delta_angle) < decay_fov_rad / 2.0)

        # [核心优化] 动态计算衰减量
        # Total Decay = 基础 + (线速度 * 因子) + (角速度 * 因子)
        # 旋转得越快，清除力度越大！
        dynamic_decay = self.base_decay_rate + \
                        (self.current_linear_speed * self.velocity_decay_factor) + \
                        (self.current_yaw_rate * self.angular_decay_factor)
                        
        decay_amount = dynamic_decay * dt
        
        roi_grid = self.occupancy_grid[min_map_y:max_map_y, min_map_x:max_map_x]
        roi_grid[fov_mask] -= decay_amount
        roi_grid[fov_mask] = np.clip(roi_grid[fov_mask], 0, self.max_occupancy)

    def dilate_mask(self, mask, radius):
        if radius <= 0: return mask
        dilated = mask.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0: continue
                rolled = np.roll(mask, shift=(dy, dx), axis=(0, 1))
                if dy > 0: rolled[:dy, :] = False
                elif dy < 0: rolled[dy:, :] = False
                if dx > 0: rolled[:, :dx] = False
                elif dx < 0: rolled[:, dx:] = False
                dilated |= rolled
        return dilated

    def quaternion_to_matrix(self, x, y, z, w):
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])
        return R

    def get_euler_from_quaternion(self, x, y, z, w):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def publish_map(self, stamp):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = stamp
        grid_msg.header.frame_id = self.map_frame
        grid_msg.info.resolution = self.map_res
        grid_msg.info.width = self.grid_shape[1]
        grid_msg.info.height = self.grid_shape[0]
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        grid_msg.info.origin.position.z = self.map_z_level
        grid_msg.info.origin.orientation.w = 1.0
        
        final_data = np.clip(self.occupancy_grid, 0, 100)
        final_data = np.round(final_data).astype(np.int8)
        
        grid_msg.data = final_data.flatten().tobytes()
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        node = DirectSonarMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass