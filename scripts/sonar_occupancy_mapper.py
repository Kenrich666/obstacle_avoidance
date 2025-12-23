#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
import math

class SonarOccupancyMapper:
    def __init__(self):
        rospy.init_node('sonar_occupancy_mapper', anonymous=True)
        
        # --- 参数配置 ---
        self.map_frame = "world"
        self.robot_base_frame = "rexrov2/base_link" # 机器人基座 Frame
        
        self.map_res = 0.2
        # 地图尺寸 (300m x 300m)
        self.map_width_m = 300.0
        self.map_height_m = 300.0
        self.map_origin_x = -150.0
        self.map_origin_y = -150.0
        
        # 声纳传感器参数
        self.sonar_max_range = 16.0 
        self.sonar_fov_rad = np.deg2rad(95.0) 
        
        # 贝叶斯/计数更新参数
        self.hit_increment = 30.0
        self.max_occupancy = 100.0
        
        # 衰减参数 (仅在可视区域内衰减，速度快一点)
        self.base_decay_rate = 5.0  
        self.velocity_decay_factor = 1.0 
        
        # 插值/膨胀参数
        self.dilation_radius = 1
        
        # 姿态过滤阈值 (度)
        self.max_tilt_deg = 5.0
        
        # --- 内部变量 ---
        self.grid_shape = (int(self.map_height_m / self.map_res), 
                           int(self.map_width_m / self.map_res))
        
        self.occupancy_grid = np.zeros(self.grid_shape, dtype=np.float32)
        self.current_speed = 0.0
        
        self.last_update_time = rospy.Time.now()
        
        # --- TF 监听器 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- 订阅与发布 ---
        rospy.Subscriber('/sonar_cloud_body', PointCloud2, self.cloud_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/rexrov2/pose_gt', Odometry, self.odom_callback, queue_size=1)
        
        self.map_pub = rospy.Publisher('/projected_sonar_map', OccupancyGrid, queue_size=1)
        
        rospy.loginfo(f"Sonar Mapper Started. (Base Frame: {self.robot_base_frame})")

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = np.sqrt(vx**2 + vy**2)

    def cloud_callback(self, cloud_msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        if dt < 0: dt = 0.0
        self.last_update_time = current_time

        # =========================================================
        # 步骤 1: 严格姿态检查 (Rigorous Attitude Check)
        # =========================================================
        try:
            # 显式查询机器人基座(Base Link)的变换，而非声纳的变换
            base_trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.robot_base_frame, 
                cloud_msg.header.stamp, 
                rospy.Duration(0.1)
            )
            
            bq_x = base_trans.transform.rotation.x
            bq_y = base_trans.transform.rotation.y
            bq_z = base_trans.transform.rotation.z
            bq_w = base_trans.transform.rotation.w
            
            # 计算基座的 Roll 和 Pitch
            base_roll, base_pitch, _ = self.get_euler_from_quaternion(bq_x, bq_y, bq_z, bq_w)
            
            # 检查是否超过阈值
            max_tilt_rad = np.deg2rad(self.max_tilt_deg)
            if abs(base_roll) > max_tilt_rad or abs(base_pitch) > max_tilt_rad:
                # 姿态不稳，跳过更新，防止地图模糊
                # 此时仅发布旧地图以保持 Topic 活跃
                self.publish_map(cloud_msg.header.stamp)
                return

        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException):
            # 如果查不到基座姿态，为安全起见跳过
            return

        # =========================================================
        # 步骤 2: 获取声纳传感器位姿 (用于数据投影)
        # =========================================================
        try:
            sensor_trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                cloud_msg.header.frame_id, 
                cloud_msg.header.stamp, 
                rospy.Duration(0.1)
            )
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException):
            return

        # 提取传感器参数
        t_x = sensor_trans.transform.translation.x
        t_y = sensor_trans.transform.translation.y
        t_z = sensor_trans.transform.translation.z
        
        q_x = sensor_trans.transform.rotation.x
        q_y = sensor_trans.transform.rotation.y
        q_z = sensor_trans.transform.rotation.z
        q_w = sensor_trans.transform.rotation.w
        
        # 计算传感器的 Yaw (用于 FOV 扇区计算)
        _, _, sensor_yaw = self.get_euler_from_quaternion(q_x, q_y, q_z, q_w)

        # =========================================================
        # 步骤 3: 处理点云 (Hit Update)
        # =========================================================
        gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        points_sensor = np.array(list(gen), dtype=np.float32)
        
        # 计算旋转矩阵
        rotation_matrix = self.quaternion_to_matrix(q_x, q_y, q_z, q_w)
        translation = np.array([t_x, t_y, t_z])

        if points_sensor.size > 0:
            # 坐标转换: Sensor -> World
            points_world = np.dot(points_sensor, rotation_matrix.T) + translation
            
            # 映射到网格索引
            j_indices = ((points_world[:, 0] - self.map_origin_x) / self.map_res).astype(int)
            i_indices = ((points_world[:, 1] - self.map_origin_y) / self.map_res).astype(int)
            
            # 过滤有效范围
            valid_mask = (i_indices >= 0) & (i_indices < self.grid_shape[0]) & \
                         (j_indices >= 0) & (j_indices < self.grid_shape[1])
            
            valid_i = i_indices[valid_mask]
            valid_j = j_indices[valid_mask]

            if len(valid_i) > 0:
                # 生成 Hit Mask 并膨胀
                current_hits = np.zeros(self.grid_shape, dtype=bool)
                current_hits[valid_i, valid_j] = True
                dilated_hits = self.dilate_mask(current_hits, self.dilation_radius)
                
                # 增加概率
                self.occupancy_grid[dilated_hits] += self.hit_increment
                self.occupancy_grid[dilated_hits] = np.clip(self.occupancy_grid[dilated_hits], 0, self.max_occupancy)

        # =========================================================
        # 步骤 4: 仅在声纳 FOV 范围内应用衰减 (Miss/Decay Update)
        # =========================================================
        self.apply_fov_decay(dt, t_x, t_y, sensor_yaw)

        # 发布地图
        self.publish_map(cloud_msg.header.stamp)

    def apply_fov_decay(self, dt, sensor_x, sensor_y, sensor_yaw):
        if dt <= 0: return

        # 1. 确定 ROI (Region of Interest)
        range_indices = int(self.sonar_max_range / self.map_res) + 1
        cx = int((sensor_x - self.map_origin_x) / self.map_res)
        cy = int((sensor_y - self.map_origin_y) / self.map_res)

        min_x = max(0, cx - range_indices)
        max_x = min(self.grid_shape[1], cx + range_indices)
        min_y = max(0, cy - range_indices)
        max_y = min(self.grid_shape[0], cy + range_indices)

        if min_x >= max_x or min_y >= max_y:
            return

        # 2. 生成 ROI 内的坐标网格
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        world_x = grid_x * self.map_res + self.map_origin_x
        world_y = grid_y * self.map_res + self.map_origin_y
        
        dx = world_x - sensor_x
        dy = world_y - sensor_y

        # 3. 计算极坐标
        dist_sq = dx**2 + dy**2
        angles = np.arctan2(dy, dx)

        # 计算角度差 (归一化到 -pi ~ pi)
        delta_angle = angles - sensor_yaw
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi

        # 4. 生成 Mask (扇形区域)
        decay_max_range = self.sonar_max_range + 1.0
        decay_fov_rad = self.sonar_fov_rad + np.deg2rad(5.0)
        fov_mask = (dist_sq < decay_max_range**2) & \
                   (np.abs(delta_angle) < decay_fov_rad / 2.0)

        # 5. 应用衰减
        decay_amount = (self.base_decay_rate + self.current_speed * self.velocity_decay_factor) * dt
        
        roi_grid = self.occupancy_grid[min_y:max_y, min_x:max_x]
        
        # 仅对掩码区域减去衰减值 (原位操作)
        roi_grid[fov_mask] -= decay_amount
        
        roi_grid[fov_mask] = np.clip(roi_grid[fov_mask], 0, self.max_occupancy)

    # --- 辅助函数 ---
    def dilate_mask(self, mask, radius):
        if radius <= 0: return mask
        dilated = mask.copy()
        rows, cols = mask.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0: continue
                s_y0, s_y1 = max(0, -dy), min(rows, rows - dy)
                s_x0, s_x1 = max(0, -dx), min(cols, cols - dx)
                d_y0, d_y1 = max(0, dy), min(rows, rows + dy)
                d_x0, d_x1 = max(0, dx), min(cols, cols + dx)
                dilated[d_y0:d_y1, d_x0:d_x1] |= mask[s_y0:s_y1, s_x0:s_x1]
        return dilated

    def quaternion_to_matrix(self, x, y, z, w):
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])
        return R

    def get_euler_from_quaternion(self, x, y, z, w):
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        # Yaw
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
        grid_msg.info.origin.position.z = -76.0
        grid_msg.info.origin.orientation.w = 1.0
        
        # 安全转换：float32 -> int8
        grid_msg.data = self.occupancy_grid.astype(np.int8).flatten().tobytes()
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        node = SonarOccupancyMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass