#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from marine_acoustic_msgs.msg import ProjectedSonarImage
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

class SonarToCloudNode:
    def __init__(self):
        rospy.init_node('sonar_to_cloud_node', anonymous=True)
        
        # 订阅声纳图像话题
        rospy.Subscriber('/rexrov2/blueview_p900/sonar_image_raw', ProjectedSonarImage, self.sonar_callback, queue_size=1)
        
        # 发布点云话题
        self.pc_pub = rospy.Publisher('/sonar_cloud_body', PointCloud2, queue_size=1)
        
        # 参数配置
        self.target_frame = "blueview_p900_link"  # 用户指定坐标系
        self.fov_deg = 90.0                         # 用户指定视场角
        self.beam_count_target = 512                # 用户指定波束数
        self.min_range = 1.0                        # 用户指定排除1m内噪声
        self.intensity_threshold = 5               # 强度阈值(根据实际效果调整，原代码为5可能过低)
        
        # 预计算角度缓存 (如果波束数固定，可减少重复计算)
        self.cached_angles = None
        
        rospy.loginfo("声纳点云转换节点已启动 (Vectorized). Frame: %s", self.target_frame)

    def sonar_callback(self, data):
        """
        回调函数：处理声纳图像并转换为点云
        使用 Numpy 向量化操作以极高效率处理数据
        """
        # 1. 获取基础数据
        ranges = np.array(data.ranges, dtype=np.float32) # [R]
        raw_image = np.frombuffer(data.image.data, dtype=np.uint8)
        
        n_ranges = len(ranges)
        if n_ranges == 0:
            return

        # 2. 计算/验证波束数 (Beams)
        # 图像通常是 Ranges * Beams 大小
        if len(raw_image) % n_ranges != 0:
            rospy.logwarn("图像数据长度与距离分辨率不匹配，跳过帧")
            return
            
        n_beams = int(len(raw_image) / n_ranges)
        
        # 重塑图像矩阵：行=Range(距离), 列=Beam(角度)
        try:
            # 注意：需根据实际声纳SDK确认是 (Rows, Cols) 还是 (Cols, Rows)
            # 此处假设数据排列为：先按Beam及其Range填充 (Range行, Beam列)
            matrix_data = raw_image.reshape((n_ranges, n_beams))
        except ValueError:
            return

        # 3. 准备角度数据 (Theta)
        # 如果波束数发生变化，或者缓存尚未初始化
        if self.cached_angles is None or n_beams != self.last_beam_count:
            fov_rad = np.deg2rad(self.fov_deg)
            # 生成从 +FOV/2 (左) 到 -FOV/2 (右) 的角度，解决镜像问题
            self.cached_angles = np.linspace(fov_rad/2, -fov_rad/2, n_beams)
            self.last_beam_count = n_beams
            
        angles = self.cached_angles

        # 4. 数据过滤 (核心优化部分)
        
        # 4.1 距离过滤：排除 1m 内的噪声
        # 创建距离掩码 (Boolean Mask)
        range_mask = ranges > self.min_range
        
        # 4.2 强度过滤
        # 为了效率，先切片掉近距离的数据，减少后续计算量
        # 获取有效距离部分的数据矩阵
        valid_matrix = matrix_data[range_mask, :]
        valid_ranges = ranges[range_mask]
        
        if valid_matrix.size == 0:
            return

        # 创建强度掩码
        intensity_mask = valid_matrix > self.intensity_threshold
        
        # 如果没有点满足条件，直接返回
        if not np.any(intensity_mask):
            return

        # 5. 坐标计算 (Polar -> Cartesian)
        # 利用 Meshgrid 生成对应的 R 和 Theta 矩阵
        # R_grid 对应每个像素的距离, Theta_grid 对应每个像素的角度
        # indexing='ij' 表示第一个维度是行(Range)，第二个维度是列(Beam)
        R_grid, Theta_grid = np.meshgrid(valid_ranges, angles, indexing='ij')
        
        # 应用强度掩码，只提取有效点的 R 和 Theta
        # 这步操作将 2D 矩阵扁平化为 1D 数组
        final_r = R_grid[intensity_mask]
        final_theta = Theta_grid[intensity_mask]
        final_intensity = valid_matrix[intensity_mask] # 可选：如果需要发布强度
        
        # 极坐标转笛卡尔坐标
        # ROS坐标系: X向前, Y向左, Z向上
        X = final_r * np.cos(final_theta)
        Y = final_r * np.sin(final_theta)
        Z = np.zeros_like(X) # 2D声纳 Z=0
        
        # 6. 构建点云消息
        # 将 X, Y, Z 堆叠成 (N, 3) 数组
        points_3d = np.column_stack((X, Y, Z))
        
        # 转换为 float32 以匹配 PointCloud2 格式
        points_3d = points_3d.astype(np.float32)
        
        # 发布
        self.publish_point_cloud(points_3d, data.header.stamp)

    def publish_point_cloud(self, points, stamp):
        header = Header()
        header.stamp = stamp # 保持与声纳图像相同的时间戳
        header.frame_id = self.target_frame
        
        # 使用 pc2.create_cloud_xyz32 比手动循环快得多
        # 它直接处理 numpy 数组
        pc_msg = pc2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc_msg)

if __name__ == '__main__':
    try:
        node = SonarToCloudNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass