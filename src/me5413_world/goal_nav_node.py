#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatusArray

class NavigationController:
    def __init__(self):
        rospy.init_node('goal_nav_node', anonymous=True)
        
        # 初始化发布者
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # 初始化订阅者 - 监听导航状态
        self.status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)
        
        # 导航目标点列表
        self.goals = [
            # 第一个目标点
            {'x': 16.0, 'y': -22.0, 'z': 0.0, 'yaw': math.pi},
            # 第二个目标点
            {'x': 8.9, 'y': -11.6, 'z': 0.0, 'yaw': 0.0}
        ]
        
        # 当前目标点索引
        self.current_goal_index = 0
        
        # 标记导航是否完成
        self.navigation_completed = False
        
        # 等待节点初始化
        rospy.sleep(2)
        rospy.loginfo("Navigation controller initialized")

    def publish_initial_pose(self):
        """发布初始位姿"""
        initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        rospy.sleep(1)  # 确保发布器初始化完成

        # 设置初始位姿
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = "map"
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.pose.pose.position.x = 0.0  # 初始位置 x
        initial_pose.pose.pose.position.y = 0.0  # 初始位置 y
        initial_pose.pose.pose.position.z = 0.0  # 初始位置 z

        yaw = 0.0  # 初始朝向（弧度）
        initial_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        initial_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # 设置协方差矩阵
        initial_pose.pose.covariance = [0.0] * 36
        initial_pose.pose.covariance[0] = 0.25  # x 方向的协方差
        initial_pose.pose.covariance[7] = 0.25  # y 方向的协方差
        initial_pose.pose.covariance[35] = 0.06853891945200942  # yaw 的协方差

        initial_pose_pub.publish(initial_pose)
        rospy.loginfo("Published initial pose")

    def publish_next_goal(self):
        """发布下一个导航目标点"""
        if self.current_goal_index >= len(self.goals):
            rospy.loginfo("All navigation goals completed")
            self.navigation_completed = True
            return
        
        goal_data = self.goals[self.current_goal_index]
        
        # 创建目标点消息
        goal = PoseStamped()
        goal.header.frame_id = "map"  
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = goal_data['x']
        goal.pose.position.y = goal_data['y']
        goal.pose.position.z = goal_data['z']

        yaw = goal_data['yaw']
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # 发布目标点
        rospy.loginfo(f"Publishing goal {self.current_goal_index + 1}: x={goal_data['x']}, y={goal_data['y']}")
        self.goal_pub.publish(goal)
        
    def status_callback(self, msg):
        """处理导航状态反馈"""
        if not msg.status_list:
            return
            
        # 获取最新状态
        status = msg.status_list[-1].status
        
        # 状态为3表示目标已到达
        if status == 3:  # SUCCEEDED
            rospy.loginfo(f"Reached goal {self.current_goal_index + 1}")
            
            # 如果是最后一个目标点
            if self.current_goal_index == len(self.goals) - 1:
                rospy.loginfo("Reached final destination, navigation completed")
                self.navigation_completed = True
            else:
                # 前往下一个目标点
                self.current_goal_index += 1
                rospy.sleep(2)  # 稍作等待
                self.publish_next_goal()
                
    def start_navigation(self):
        """开始导航任务"""
        # 可选：发布初始位姿
        # self.publish_initial_pose()
        
        # 发布第一个目标点
        self.publish_next_goal()
        
        # 进入主循环
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            if self.navigation_completed:
                rospy.loginfo("Navigation mission complete!")
                break
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = NavigationController()
        controller.start_navigation()
    except rospy.ROSInterruptException:
        pass