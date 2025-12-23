#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import tf
from sensor_msgs.msg import PointCloud, PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

class ConfidenceGridMapper:
    def __init__(self):
        rospy.init_node('confidence_grid_mapper')

        # --- 1. 参数配置 ---
        # 地图参数
        self.map_frame = rospy.get_param('~map_frame', 'world')    # 固定地图坐标系
        self.grid_res = rospy.get_param('~resolution', 0.1)        # 分辨率 (m/cell)
        self.map_width_m = rospy.get_param('~width', 300.0)        # 地图实际宽度 (m)
        self.map_height_m = rospy.get_param('~height', 300.0)      # 地图实际高度 (m)
        
        # 贝叶斯/计数更新参数 (按照你的要求)
        self.p_init = 50          # 初始概率 (未知)
        self.p_hit = rospy.get_param('~prob_hit', 30)              # 击中增加量
        self.p_decay = rospy.get_param('~prob_decay', 5)           # 全局衰减量
        self.val_min = 0          # 概率下限
        self.val_max = 100        # 概率上限

        # 形态学处理参数
        self.blur_ksize = rospy.get_param('~blur_ksize', 5)        # 高斯核大小 (必须是奇数)
        self.blur_sigma = rospy.get_param('~blur_sigma', 1.5)      # 高斯标准差

        # --- 2. 初始化网格 ---
        self.grid_w = int(self.map_width_m / self.grid_res)
        self.grid_h = int(self.map_height_m / self.grid_res)
        
        # 设定地图原点：默认地图中心对应世界坐标 (0,0)
        # 如果你想让地图覆盖特定区域，可以修改这里的 origin_x/y
        self.origin_x = -self.map_width_m / 2.0
        self.origin_y = -self.map_height_m / 2.0

        # 使用 float32 进行计算以保持精度，发布时转 int8
        self.grid_data = np.full((self.grid_h, self.grid_w), self.p_init, dtype=np.float32)

        # --- 3. ROS 接口 ---
        self.tf_listener = tf.TransformListener()
        
        # 订阅点云 (自动适配 PointCloud 或 PointCloud2)
        # 假设上游 topic 为 /sonar_cloud_body 或 /sonar_to_cloud/cloud
        self.sub_pc1 = rospy.Subscriber('sonar_cloud', PointCloud, self.pc1_callback)
        self.sub_pc2 = rospy.Subscriber('sonar_cloud', PointCloud2, self.pc2_callback)
        
        # 发布处理后的栅格地图
        self.pub_grid = rospy.Publisher('/confidence_grid', OccupancyGrid, queue_size=1)

        # 定时器：控制衰减和发布的频率 (例如 10Hz)
        self.update_timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        rospy.loginfo("Confidence Grid Mapper Initialized. Map Size: {}x{}".format(self.grid_w, self.grid_h))

    def world_to_grid(self, wx, wy):
        """ 将世界坐标转换为网格索引 """
        gx = int((wx - self.origin_x) / self.grid_res)
        gy = int((wy - self.origin_y) / self.grid_res)
        return gx, gy

    def process_points(self, points_list, frame_id, stamp):
        """ 核心处理逻辑：坐标转换 + 击中更新 """
        if not points_list:
            return

        try:
            # 等待并获取 TF 变换
            self.tf_listener.waitForTransform(self.map_frame, frame_id, stamp, rospy.Duration(0.2))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame, frame_id, stamp)
            mat44 = self.tf_listener.fromTranslationRotation(trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF Error: %s" % str(e))
            return

        # 批量坐标转换 (简化版，假设 points_list 是 Nx3 数组)
        # 为提高效率，这里逐点处理 (Python中处理大点云建议用 numpy 矩阵乘法，但声纳点通常不多)
        
        hit_indices_x = []
        hit_indices_y = []

        for p in points_list:
            # p 是 (x, y, z)
            # 应用变换矩阵
            px = mat44[0,0]*p[0] + mat44[0,1]*p[1] + mat44[0,2]*p[2] + mat44[0,3]
            py = mat44[1,0]*p[0] + mat44[1,1]*p[1] + mat44[1,2]*p[2] + mat44[1,3]
            
            gx, gy = self.world_to_grid(px, py)

            # 检查边界
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                hit_indices_x.append(gx)
                hit_indices_y.append(gy)

        # --- 贝叶斯/计数更新 (Hit) ---
        if hit_indices_x:
            # 利用 Numpy 高级索引进行批量更新
            # 注意：grid[y, x]
            self.grid_data[hit_indices_y, hit_indices_x] += self.p_hit
            
            # 限制上限
            np.clip(self.grid_data, self.val_min, self.val_max, out=self.grid_data)

    def pc1_callback(self, msg):
        """ 处理 sensor_msgs/PointCloud """
        points = [(p.x, p.y, p.z) for p in msg.points]
        self.process_points(points, msg.header.frame_id, msg.header.stamp)

    def pc2_callback(self, msg):
        """ 处理 sensor_msgs/PointCloud2 """
        # 使用生成器读取点
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = list(gen)
        self.process_points(points, msg.header.frame_id, msg.header.stamp)

    def timer_callback(self, event):
        """ 
        定时任务：
        1. 全局衰减 (Miss/Decay)
        2. 形态学处理 (Blur)
        3. 发布消息
        """
        # --- 1. 全局衰减 ---
        # 所有大于0的格子减去衰减量 (模拟记忆淡去)
        # 使用 mask 避免对已经是 0 的区域做无用减法
        mask = self.grid_data > self.val_min
        self.grid_data[mask] -= self.p_decay
        
        # 再次 clip 保证不越界 (主要是下限)
        np.clip(self.grid_data, self.val_min, self.val_max, out=self.grid_data)

        # --- 2. 形态学处理 (产生梯度) ---
        # 复制一份数据进行模糊，不影响原始 grid_data 的计数逻辑
        # 必须转为 uint8 或 float32 才能被 cv2 处理
        current_view = self.grid_data.astype(np.float32)

        # 高斯模糊
        k = self.blur_ksize
        blurred = cv2.GaussianBlur(current_view, (k, k), self.blur_sigma)

        # --- 3. 发布 OccupancyGrid ---
        self.publish_map(blurred)

    def publish_map(self, grid_to_pub):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        
        msg.info.resolution = self.grid_res
        msg.info.width = self.grid_w
        msg.info.height = self.grid_h
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0
        msg.info.origin.orientation.w = 1.0

        # 数据转换：Costmap 接受 int8 [-1, 100]
        # 我们这里的数据是 0-100 的 float
        data_int8 = grid_to_pub.astype(np.int8)
        
        # 展平并赋值
        msg.data = data_int8.flatten().tolist()
        
        self.pub_grid.publish(msg)

if __name__ == '__main__':
    try:
        ConfidenceGridMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass