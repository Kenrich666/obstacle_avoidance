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
        self.robot_base_frame = "rexrov2/base_link"
        
        self.map_res = 0.2
        self.map_width_m = 300.0
        self.map_height_m = 300.0
        self.map_origin_x = -150.0
        self.map_origin_y = -150.0
        
        # [修改] 问题3: 移除硬编码 Z 轴，设置为 0.0 或参数化
        self.map_z_level = 0.0 
        
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
        
        # [修改] 问题2: 初始化为 None，防止第一帧 dt 过大
        self.last_update_time = None
        
        # [修改] 问题6: 性能优化 - 预计算查找表 (Cache)
        self._init_precomputed_tables()

        # --- TF 监听器 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- 订阅与发布 ---
        rospy.Subscriber('/sonar_cloud_body', PointCloud2, self.cloud_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/rexrov2/pose_gt', Odometry, self.odom_callback, queue_size=1)
        
        self.map_pub = rospy.Publisher('/projected_sonar_map', OccupancyGrid, queue_size=1)
        
        rospy.loginfo(f"Sonar Mapper Started. Map Size: {self.grid_shape}")

    def _init_precomputed_tables(self):
        """
        [修改] 问题6: 预计算相对坐标和角度，避免在回调中进行昂贵的 meshgrid 和 arctan2 运算
        """
        # 计算 ROI 半径对应的网格数
        self.range_indices = int(self.sonar_max_range / self.map_res) + 1
        size = 2 * self.range_indices + 1
        
        # 生成相对中心的网格 (像素坐标)
        x_range = np.arange(-self.range_indices, self.range_indices + 1)
        y_range = np.arange(-self.range_indices, self.range_indices + 1)
        self.cache_grid_x, self.cache_grid_y = np.meshgrid(x_range, y_range)
        
        # 转换为相对物理坐标
        rel_world_x = self.cache_grid_x * self.map_res
        rel_world_y = self.cache_grid_y * self.map_res
        
        # 预计算距离平方 (用于距离判断)
        self.cache_dist_sq = rel_world_x**2 + rel_world_y**2
        
        # 预计算相对角度 (用于 FOV 判断)
        self.cache_angles = np.arctan2(rel_world_y, rel_world_x)
        
        rospy.loginfo("Performance cache initialized.")

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = np.sqrt(vx**2 + vy**2)

    def cloud_callback(self, cloud_msg):
        current_time = rospy.Time.now()
        
        # [修改] 问题2: 处理第一帧逻辑
        if self.last_update_time is None:
            self.last_update_time = current_time
            # 第一帧只做初始化，不计算 dt，避免巨大的 decay
            return

        dt = (current_time - self.last_update_time).to_sec()
        if dt < 0: dt = 0.0
        self.last_update_time = current_time

        # =========================================================
        # 步骤 1: 严格姿态检查
        # =========================================================
        try:
            # 显式查询机器人基座(Base Link)的变换，而非声纳的变换
            # [修改] 问题4: 增加超时时间到 0.5s，提高系统高负载下的鲁棒性
            base_trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.robot_base_frame, 
                cloud_msg.header.stamp, 
                rospy.Duration(0.5) 
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

        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"TF Lookup Failed (Base): {e}")
            return

        # =========================================================
        # 步骤 2: 获取声纳传感器位姿
        # =========================================================
        try:
            # [修改] 问题4: 增加超时时间
            sensor_trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                cloud_msg.header.frame_id, 
                cloud_msg.header.stamp, 
                rospy.Duration(0.5)
            )
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"TF Lookup Failed (Sensor): {e}")
            return

        # 提取传感器参数
        t_x = sensor_trans.transform.translation.x
        t_y = sensor_trans.transform.translation.y
        t_z = sensor_trans.transform.translation.z
        
        q_x = sensor_trans.transform.rotation.x
        q_y = sensor_trans.transform.rotation.y
        q_z = sensor_trans.transform.rotation.z
        q_w = sensor_trans.transform.rotation.w
        
        _, _, sensor_yaw = self.get_euler_from_quaternion(q_x, q_y, q_z, q_w)

        # =========================================================
        # 步骤 3: 处理点云 (Hit Update)
        # =========================================================
        gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        points_sensor = np.array(list(gen), dtype=np.float32)
        
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
                current_hits = np.zeros(self.grid_shape, dtype=bool)
                current_hits[valid_i, valid_j] = True
                dilated_hits = self.dilate_mask(current_hits, self.dilation_radius)
                
                self.occupancy_grid[dilated_hits] += self.hit_increment
                self.occupancy_grid[dilated_hits] = np.clip(self.occupancy_grid[dilated_hits], 0, self.max_occupancy)

        # =========================================================
        # 步骤 4: 衰减更新 (使用优化后的逻辑，保留原始衰减算法)
        # =========================================================
        self.apply_fov_decay_optimized(dt, t_x, t_y, sensor_yaw)

        self.publish_map(cloud_msg.header.stamp)

    def apply_fov_decay_optimized(self, dt, sensor_x, sensor_y, sensor_yaw):
        """
        [修改] 问题6: 使用预计算的 Cache 进行快速衰减计算
        逻辑保持不变：在 FOV 范围内进行衰减。
        """
        if dt <= 0: return

        # 1. 计算传感器在 Grid 中的整数索引
        cx = int((sensor_x - self.map_origin_x) / self.map_res)
        cy = int((sensor_y - self.map_origin_y) / self.map_res)

        # 2. 确定裁剪区域 (处理地图边界)
        # 这里的 min/max 是相对于 Cache 数组的索引
        pad = self.range_indices
        
        # 计算在地图上的绝对范围
        min_map_x = cx - pad
        max_map_x = cx + pad + 1
        min_map_y = cy - pad
        max_map_y = cy + pad + 1
        
        # 计算在 Cache 上的切片范围
        cache_min_x, cache_max_x = 0, self.cache_grid_x.shape[1]
        cache_min_y, cache_max_y = 0, self.cache_grid_y.shape[0]

        # 边界裁剪逻辑
        if min_map_x < 0:
            cache_min_x = -min_map_x
            min_map_x = 0
        if min_map_y < 0:
            cache_min_y = -min_map_y
            min_map_y = 0
        if max_map_x > self.grid_shape[1]:
            diff = max_map_x - self.grid_shape[1]
            cache_max_x -= diff
            max_map_x = self.grid_shape[1]
        if max_map_y > self.grid_shape[0]:
            diff = max_map_y - self.grid_shape[0]
            cache_max_y -= diff
            max_map_y = self.grid_shape[0]

        # 如果完全在地图外，直接返回
        if min_map_x >= max_map_x or min_map_y >= max_map_y:
            return

        # 3. 提取 Cache 切片 (避免重复计算 distance 和 angle)
        roi_dist_sq = self.cache_dist_sq[cache_min_y:cache_max_y, cache_min_x:cache_max_x]
        roi_angles = self.cache_angles[cache_min_y:cache_max_y, cache_min_x:cache_max_x]

        # 4. 计算角度差 (只需要做减法)
        # 将传感器 Yaw 减去，得到相对于传感器朝向的角度
        delta_angle = roi_angles - sensor_yaw
        delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi

        # 5. 生成 Mask
        decay_max_range = self.sonar_max_range + 1.0
        decay_fov_rad = self.sonar_fov_rad + np.deg2rad(5.0)
        
        fov_mask = (roi_dist_sq < decay_max_range**2) & \
                   (np.abs(delta_angle) < decay_fov_rad / 2.0)

        # 6. 应用衰减 (直接操作切片)
        decay_amount = (self.base_decay_rate + self.current_speed * self.velocity_decay_factor) * dt
        
        # 获取地图上的 ROI 视图
        roi_grid = self.occupancy_grid[min_map_y:max_map_y, min_map_x:max_map_x]
        
        # 应用更新
        roi_grid[fov_mask] -= decay_amount
        roi_grid[fov_mask] = np.clip(roi_grid[fov_mask], 0, self.max_occupancy)

    # --- 辅助函数保持不变 ---
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
        # [修改] 问题3: 使用配置好的 Z 高度
        grid_msg.info.origin.position.z = self.map_z_level
        grid_msg.info.origin.orientation.w = 1.0
        
        # [修改] 问题5: 显式 round 和 clip，确保类型转换安全
        # 1. Clip 限制在 0-100
        # 2. Round 四舍五入，而不是默认的 floor
        # 3. 转换为 int8
        final_data = np.clip(self.occupancy_grid, 0, 100)
        final_data = np.round(final_data).astype(np.int8)
        
        grid_msg.data = final_data.flatten().tobytes()
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        node = SonarOccupancyMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass