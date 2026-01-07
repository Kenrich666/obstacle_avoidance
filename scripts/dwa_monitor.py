#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import math
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class DWAMonitor:
    def __init__(self):
        rospy.init_node('dwa_data_monitor', anonymous=True)
        
        self.last_cmd = None
        self.last_target = None
        self.last_best_traj = None
        self.last_candidates = None
        self.last_odom = None
        
        self.last_cmd_time = 0
        self.last_target_time = 0
        self.last_traj_time = 0
        self.last_cand_time = 0
        self.last_odom_time = 0

        # 订阅
        rospy.Subscriber('/rexrov2/cmd_vel_move_base', Twist, self.cmd_callback)
        rospy.Subscriber('/dwa_current_target', Marker, self.target_callback)
        rospy.Subscriber('/dwa_best_traj', Path, self.best_traj_callback)
        rospy.Subscriber('/dwa_trajectories', MarkerArray, self.candidate_callback)
        # 增加订阅里程计
        rospy.Subscriber('/rexrov2/pose_gt', Odometry, self.odom_callback)

        rospy.loginfo("DWA Monitor (Coordinate Fixed) started...")

    def cmd_callback(self, msg):
        self.last_cmd = msg
        self.last_cmd_time = time.time()

    def target_callback(self, msg):
        self.last_target = msg
        self.last_target_time = time.time()

    def best_traj_callback(self, msg):
        self.last_best_traj = msg
        self.last_traj_time = time.time()

    def candidate_callback(self, msg):
        self.last_candidates = msg
        self.last_cand_time = time.time()
    
    def odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_time = time.time()

    def print_status(self):
        now = time.time()
        print("\n" + "="*60)
        print(f"{TerminalColors.HEADER}>>> DWA Planner 状态监控 (已修正坐标系) Time: {now:.2f}{TerminalColors.ENDC}")

        # --- 1. 里程计与速度计算 ---
        current_yaw = 0.0
        v_body_x = 0.0
        v_body_y = 0.0
        
        if self.last_odom and (now - self.last_odom_time) < 2.0:
            pos = self.last_odom.pose.pose.position
            ori = self.last_odom.pose.pose.orientation
            
            # 获取 Yaw
            _, _, current_yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            
            # 获取世界坐标系速度
            v_world_x = self.last_odom.twist.twist.linear.x
            v_world_y = self.last_odom.twist.twist.linear.y
            
            # [关键] 坐标变换: World -> Body
            # V_body = R_transpose * V_world
            # R = [[cos, -sin], [sin, cos]]
            v_body_x = math.cos(current_yaw) * v_world_x + math.sin(current_yaw) * v_world_y
            v_body_y = -math.sin(current_yaw) * v_world_x + math.cos(current_yaw) * v_world_y
            
            print(f"[{TerminalColors.OKBLUE}ROBOT STATE{TerminalColors.ENDC}]")
            print(f"  Pos: ({pos.x:.2f}, {pos.y:.2f}) | Yaw: {math.degrees(current_yaw):.1f}°")
            print(f"  Speed (Body): Fwd={v_body_x:.3f} m/s, Lat={v_body_y:.3f} m/s")
        else:
            print(f"[{TerminalColors.FAIL}ROBOT STATE{TerminalColors.ENDC}] 无里程计数据")

        # --- 2. 指令 vs 实际 ---
        if self.last_cmd and (now - self.last_cmd_time) < 2.0:
            cmd_x = self.last_cmd.linear.x
            cmd_w = self.last_cmd.angular.z
            
            # 误差分析
            diff_v = cmd_x - v_body_x
            
            # 状态颜色
            status_color = TerminalColors.OKGREEN
            if abs(cmd_x) > 0.01 and v_body_x < 0.01: status_color = TerminalColors.WARNING # 有指令不动
            if v_body_x < -0.05: status_color = TerminalColors.FAIL # 倒车
            
            print(f"[{status_color}CONTROL LOOP{TerminalColors.ENDC}]")
            print(f"  CMD: {cmd_x:.2f} m/s | ACT: {v_body_x:.3f} m/s | Diff: {diff_v:.3f}")
            if abs(cmd_x) > 0.05:
                # 简单的跟随率计算
                ratio = (v_body_x / cmd_x) * 100
                print(f"  跟随率: {ratio:.1f}% (由于水阻，<100%是正常的)")
        else:
             print(f"[{TerminalColors.FAIL}CONTROL LOOP{TerminalColors.ENDC}] 无指令数据")

        # --- 3. DWA 内部数据 ---
        if self.last_target and (now - self.last_target_time) < 2.0:
             print(f"[{TerminalColors.OKGREEN}TARGET{TerminalColors.ENDC}] OK | QuatW: {self.last_target.pose.orientation.w:.2f}")
        else:
             print(f"[{TerminalColors.WARNING}TARGET{TerminalColors.ENDC}] 等待中...")

        if self.last_best_traj and (now - self.last_traj_time) < 2.0:
             print(f"[{TerminalColors.OKGREEN}PATH{TerminalColors.ENDC}] OK | Points: {len(self.last_best_traj.poses)}")
        else:
             print(f"[{TerminalColors.WARNING}PATH{TerminalColors.ENDC}] 搜索中...")

    def run(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.print_status()
            rate.sleep()

if __name__ == '__main__':
    try:
        monitor = DWAMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass