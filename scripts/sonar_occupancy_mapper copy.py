#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
# 移除了 tf2_sensor_msgs 的依赖

class SonarOccupancyMapper:
    def __init__(self):
        rospy.init_node('sonar_occupancy_mapper', anonymous=True)
        
        # --- 参数配置 ---
        self.map_frame = "world"              # 地图固定坐标系
        self.map_res = 0.2                    # 栅格分辨率 (米/格)
        self.map_width_m = 300.0              # 地图宽 (米)
        self.map_height_m = 300.0             # 地图高 (米)
        self.map_origin_x = -150.0             # 地图原点 X (米)
        self.map_origin_y = -150.0             # 地图原点 Y (米)
        
        # 贝叶斯/计数更新参数
        self.hit_increment = 40.0             # 击中增加的置信度
        self.max_occupancy = 100.0            # 最大置信度
        
        # 衰减参数
        self.base_decay = 3.0                 # 基础衰减
        self.velocity_decay_factor = 0.0      # 速度衰减系数
        
        # --- 内部变量 ---
        self.grid_shape = (int(self.map_height_m / self.map_res), 
                           int(self.map_width_m / self.map_res))
        
        self.occupancy_grid = np.zeros(self.grid_shape, dtype=np.float32)
        self.current_speed = 0.0
        
        # --- TF 监听器 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- 订阅与发布 ---
        rospy.Subscriber('/sonar_cloud_body', PointCloud2, self.cloud_callback, queue_size=1)
        rospy.Subscriber('/rexrov2/pose_gt', Odometry, self.odom_callback, queue_size=1)
        
        self.map_pub = rospy.Publisher('/projected_sonar_map', OccupancyGrid, queue_size=1)
        
        rospy.loginfo(f"Sonar Occupancy Mapper Started (Numpy Version). Map Size: {self.grid_shape}")

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = np.sqrt(vx**2 + vy**2)

    def cloud_callback(self, cloud_msg):
        # 1. 获取坐标变换 (Target Frame <- Source Frame)
        try:
            # 这里的 timeout 设置为 0 表示只查最新的，也可以稍微给一点等待时间
            # trans = self.tf_buffer.lookup_transform(self.map_frame, cloud_msg.header.frame_id, rospy.Time(0))
            # 尝试查找点云采集时刻的变换，给予 0.1s 的等待宽限
            trans = self.tf_buffer.lookup_transform(self.map_frame, cloud_msg.header.frame_id, cloud_msg.header.stamp, rospy.Duration(0.1))
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            rospy.logwarn_throttle(2, f"TF Error: {e}")
            return

        # 2. 读取点云数据 (Sensor Frame)
        # 我们需要 x, y, z 来进行 3D 旋转，即使最后只用 x, y
        gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        points_sensor = np.array(list(gen), dtype=np.float32) # Shape: (N, 3)

        if points_sensor.size == 0:
            self.apply_decay()
            self.publish_map(cloud_msg.header.stamp)
            return

        # 3. 手动应用坐标变换 (代替 do_transform_cloud)
        # 提取平移向量
        t_x = trans.transform.translation.x
        t_y = trans.transform.translation.y
        t_z = trans.transform.translation.z
        translation = np.array([t_x, t_y, t_z])

        # 提取旋转四元数并转换为旋转矩阵
        q_x = trans.transform.rotation.x
        q_y = trans.transform.rotation.y
        q_z = trans.transform.rotation.z
        q_w = trans.transform.rotation.w
        
        rotation_matrix = self.quaternion_to_matrix(q_x, q_y, q_z, q_w)

        # 变换公式: P_world = P_sensor * R^T + T
        # numpy 的 dot 是矩阵乘法。由于 points_sensor 是 (N,3)，我们需要右乘旋转矩阵的转置
        points_world = np.dot(points_sensor, rotation_matrix.T) + translation

        # 4. 空间映射: World Frame (X, Y) -> Grid Indices (Col, Row)
        # 注意：OccupancyGrid 依然是 Row-Major (Row=Y, Col=X)
        j_indices = ((points_world[:, 0] - self.map_origin_x) / self.map_res).astype(int) # X -> Col
        i_indices = ((points_world[:, 1] - self.map_origin_y) / self.map_res).astype(int) # Y -> Row
        
        # 过滤有效范围
        valid_mask = (i_indices >= 0) & (i_indices < self.grid_shape[0]) & \
                     (j_indices >= 0) & (j_indices < self.grid_shape[1])
        
        valid_i = i_indices[valid_mask]
        valid_j = j_indices[valid_mask]

        # 5. 概率更新 (Hit)
        hit_mask = np.zeros(self.grid_shape, dtype=bool)
        hit_mask[valid_i, valid_j] = True
        self.occupancy_grid[hit_mask] += self.hit_increment

        # 6. 概率衰减 (Decay) & 发布
        self.apply_decay()
        self.publish_map(cloud_msg.header.stamp)

    def quaternion_to_matrix(self, x, y, z, w):
        """
        手动将四元数转换为 3x3 旋转矩阵
        """
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])
        return R

    def apply_decay(self):
        decay = self.base_decay + (self.current_speed * self.velocity_decay_factor)
        self.occupancy_grid -= decay
        self.occupancy_grid = np.clip(self.occupancy_grid, 0, self.max_occupancy)

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
        
        flat_data = self.occupancy_grid.astype(np.int8).flatten()
        grid_msg.data = flat_data.tolist()
        
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        node = SonarOccupancyMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass