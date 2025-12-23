#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion

def send_goal():
    # 1. 初始化节点
    rospy.init_node('navigation_goal_client')
    
    # 2. 创建 Action Client，连接到 move_base
    # move_base 提供了 Action 接口，允许我们发送目标并等待结果
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    rospy.loginfo("正在等待 move_base 服务器启动...")
    client.wait_for_server()
    rospy.loginfo("move_base 服务器已连接，准备发送目标。")

    # 3. 构建目标点 (Goal)
    goal = MoveBaseGoal()
    
    # 设置坐标系：必须与 global_costmap_params.yaml 中的 global_frame 一致
    # 根据之前的配置，这里应该是 "world"
    goal.target_pose.header.frame_id = "world"
    goal.target_pose.header.stamp = rospy.Time.now()

    # 设置目标位置 (用户指定的坐标)
    goal.target_pose.pose.position.x = -20.0
    goal.target_pose.pose.position.y = 0.0
    # 注意：标准的 move_base 是 2D 导航，它通常只处理 X 和 Y。
    # 虽然这里设置了 Z=-76，但如果局部规划器只输出 planar cmd_vel (x, y, yaw)，
    # 机器人可能不会潜下去，而是保持当前深度移动到目标上方。
    goal.target_pose.pose.position.z = -76.0 
    
    # 设置目标朝向 (Quaternion)
    # w=1.0 表示没有旋转（朝向世界坐标系的 X 轴正方向）
    # 机器人移动时会自动调整朝向以对准路径，到达终点后会尝试调整回这个朝向
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = 0.0
    goal.target_pose.pose.orientation.w = 1.0

    # 4. 发送目标
    rospy.loginfo(f"发送目标点: X={goal.target_pose.pose.position.x}, Y={goal.target_pose.pose.position.y}, Z={goal.target_pose.pose.position.z}")
    client.send_goal(goal)

    # 5. 等待结果 (可选)
    # 如果你想让脚本一直等到机器人到达才结束，可以取消下面的注释
    # wait = client.wait_for_result()
    # if not wait:
    #     rospy.logerr("Action server not available!")
    # else:
    #     return client.get_result()

if __name__ == '__main__':
    try:
        send_goal()
        rospy.loginfo("目标已发送！请观察 RViz 或仿真界面。")
    except rospy.ROSInterruptException:
        pass