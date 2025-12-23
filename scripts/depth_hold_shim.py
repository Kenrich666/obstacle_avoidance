#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class DepthHoldShim:
    def __init__(self):
        rospy.init_node('depth_hold_shim')
        rospy.loginfo("DepthHoldShim node started...")

        # 参数设置
        self.target_depth = -76.0 
        self.kp_z = 2.0 
        self.current_depth = 0.0
        self.depth_received = False # 标志位：是否收到过深度信息

        # 1. 确认订阅的话题名是否正确！
        # 这里的 topic 必须与 rostopic list 中 move_base 输出的一致
        cmd_topic = '/rexrov2/cmd_vel_move_base'
        rospy.loginfo("Subscribing to move_base cmd: %s", cmd_topic)
        self.sub_cmd = rospy.Subscriber(cmd_topic, Twist, self.cmd_callback)

        # 2. 确认里程计话题！
        # 使用 rostopic list 查看你的里程计到底是 /rexrov2/pose_gt 还是其他名字
        odom_topic = '/rexrov2/pose_gt' 
        rospy.loginfo("Subscribing to odometry: %s", odom_topic)
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        
        # 3. 确认发布话题
        pub_topic = '/rexrov2/cmd_vel'
        rospy.loginfo("Publishing final cmd to: %s", pub_topic)
        self.pub_cmd = rospy.Publisher(pub_topic, Twist, queue_size=1)

    def odom_callback(self, msg):
        # 收到里程计数据
        self.current_depth = msg.pose.pose.position.z
        if not self.depth_received:
            rospy.loginfo("First Odometry received! Current depth: %.2f", self.current_depth)
            self.depth_received = True

    def cmd_callback(self, msg):
        # 收到 move_base 指令
        # 调试打印（频率可能很高，确认正常后可注释掉）
        # rospy.loginfo("Received cmd from move_base: linear.x=%.2f", msg.linear.x)

        if not self.depth_received:
            rospy.logwarn_throttle(2, "Waiting for Odometry data to perform depth hold...")
            # 即使没有深度数据，也应该转发 xy 指令，防止卡死，但 z 轴给 0
            new_cmd = Twist()
            new_cmd.linear.x = msg.linear.x
            new_cmd.linear.y = msg.linear.y
            new_cmd.angular.z = msg.angular.z
            self.pub_cmd.publish(new_cmd)
            return

        # 正常控制逻辑
        new_cmd = Twist()
        new_cmd.linear.x = msg.linear.x
        new_cmd.linear.y = msg.linear.y
        new_cmd.angular.z = msg.angular.z

        # 深度控制 P 控制器
        error = self.target_depth - self.current_depth
        z_vel = self.kp_z * error
        z_vel = max(min(z_vel, 0.5), -0.5) # 限幅
        
        new_cmd.linear.z = z_vel

        # 强制压平姿态
        new_cmd.angular.x = 0.0
        new_cmd.angular.y = 0.0

        self.pub_cmd.publish(new_cmd)

if __name__ == '__main__':
    try:
        node = DepthHoldShim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass