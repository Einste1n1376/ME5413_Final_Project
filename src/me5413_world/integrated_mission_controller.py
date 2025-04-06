#!/usr/bin/env python3
# filepath: /home/colin/ME5413_Final_Project/src/me5413_world/integrated_mission_controller.py

import rospy
import math
import subprocess
import sys
import time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatusArray
from std_msgs.msg import String, Bool

class IntegratedMissionController:
    def __init__(self):
        rospy.init_node('integrated_mission_controller', anonymous=True)
        
        # 初始化发布者
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.bridge_pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=1)
        self.recognized_number_pub = rospy.Publisher('/recognized_number', String, queue_size=1)
        
        # 初始化订阅者
        self.status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)
        self.number_sub = rospy.Subscriber('/recognized_number', String, self.number_callback)
        
        # 导航目标点列表 - 过桥前
        self.goals = [
            # 第一个目标点 - 避开第一区域障碍物
            {'x': 16.0, 'y': -22.0, 'z': 0.0, 'yaw': math.pi},
            # 第二个目标点 - 到达"河边"观察障碍物
            {'x': 10, 'y': -11.6, 'z': 0.0, 'yaw': math.pi},
        ]
        
        # 过桥后的目标点列表
        self.after_bridge_goals = [
            # 过桥后的四个目标点，用于检测不同位置的数字
            {'x': 3.5, 'y': -4.8, 'z': 0.0, 'yaw': math.pi},
            {'x': 3.5, 'y': -9.6, 'z': 0.0, 'yaw': 0.0},
            {'x': 3.5, 'y': -13.4, 'z': 0.0, 'yaw': math.pi},
            {'x': 3.5, 'y': -19.8, 'z': 0.0, 'yaw': 0.0}
        ]
        
        # 状态变量
        self.current_goal_index = 0  
        self.navigation_completed = False
        self.number_recognized = False
        self.recognized_digits = []
        self.mission_stage = "NAVIGATION"  # 任务阶段: NAVIGATION, NUMBER_IDENTIFICATION, BRIDGE_CROSSING, AFTER_BRIDGE
        self.bridge_process = None
        self.number_process = None
        self.auto_detector_process = None
        self.after_bridge_process = None
        self.reached_final_goal = False
        self.bridge_crossing_completed = False
        self.after_bridge_target_number = None  # 过桥后需要识别的数字
        self.after_bridge_completed = False  # 过桥后任务是否完成
        
        # 初始化订阅者
        self.amcl_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.current_robot_pose = None  # 用于存储当前机器人位置    

        # 等待节点初始化
        rospy.sleep(2)
        rospy.loginfo("集成任务控制器初始化完成")

    def amcl_pose_callback(self, msg):
        """存储当前机器人位置"""
        self.current_robot_pose = msg

    def publish_current_pose_as_initial(self):
        """获取当前位置并重新发布为初始位姿，以减少位置漂移"""
        try:
            initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
            rospy.sleep(0.5)  # 确保发布器初始化完成
            
            # 优先使用实际位置
            if self.current_robot_pose is not None:
                # 使用最新的实际位置
                initial_pose = PoseWithCovarianceStamped()
                initial_pose.header = self.current_robot_pose.header
                initial_pose.pose = self.current_robot_pose.pose
                
                initial_pose_pub.publish(initial_pose)
                rospy.loginfo("重新发布当前实际位置作为初始位姿")
                rospy.sleep(1)  # 等待位姿初始化
                return
            
            # 设置当前位置为初始位姿
            initial_pose = PoseWithCovarianceStamped()
            initial_pose.header.frame_id = "map"
            initial_pose.header.stamp = rospy.Time.now()
            
            # 根据当前任务阶段选择目标点
            goal_list = self.goals if self.mission_stage != "AFTER_BRIDGE" else self.after_bridge_goals
            
            # 修复：使用前一个目标点位置（即当前已到达的位置）作为当前位置
            previous_goal_index = self.current_goal_index - 1
            if previous_goal_index >= 0:
                # 使用前一个目标点的位置
                previous_goal = goal_list[previous_goal_index]
                initial_pose.pose.pose.position.x = previous_goal['x'] 
                initial_pose.pose.pose.position.y = previous_goal['y']
                initial_pose.pose.pose.position.z = 0.0
                yaw = previous_goal['yaw']
            else:
                # 如果是第一个目标点，使用起始位置（或第一个目标点）
                current_goal = goal_list[self.current_goal_index]
                initial_pose.pose.pose.position.x = current_goal['x'] 
                initial_pose.pose.pose.position.y = current_goal['y']
                initial_pose.pose.pose.position.z = 0.0
                yaw = current_goal['yaw']
            
            initial_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
            initial_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
            
            # 设置协方差矩阵
            initial_pose.pose.covariance = [0.0] * 36
            initial_pose.pose.covariance[0] = 0.25  # x方向的协方差
            initial_pose.pose.covariance[7] = 0.25  # y方向的协方差
            initial_pose.pose.covariance[35] = 0.06853891945200942  # yaw的协方差
            
            initial_pose_pub.publish(initial_pose)
            rospy.loginfo("重新发布当前位置作为初始位姿")
            rospy.sleep(1)  # 等待位姿初始化
        except Exception as e:
            rospy.logwarn(f"重新发布初始位姿失败: {e}")

    def publish_next_goal(self):
        """发布下一个导航目标点"""
        # 根据当前任务阶段选择目标点列表
        goal_list = self.goals if self.mission_stage != "AFTER_BRIDGE" else self.after_bridge_goals
        
        if self.current_goal_index >= len(goal_list):
            if self.mission_stage == "NAVIGATION":
                rospy.loginfo("导航任务阶段所有目标点已完成")
                self.navigation_completed = True
            elif self.mission_stage == "AFTER_BRIDGE":
                rospy.loginfo("过桥后任务阶段所有目标点已完成")
                self.after_bridge_completed = True
            return
        
        goal_data = goal_list[self.current_goal_index]
        
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
        stage_name = "导航" if self.mission_stage == "NAVIGATION" else "过桥后"
        rospy.loginfo(f"发布{stage_name}目标点 {self.current_goal_index + 1}: x={goal_data['x']}, y={goal_data['y']}")
        self.goal_pub.publish(goal)
        
    def status_callback(self, msg):
        """处理导航状态反馈"""
        if not msg.status_list or (self.mission_stage != "NAVIGATION" and self.mission_stage != "AFTER_BRIDGE"):
            return
            
        # 获取最新状态
        status = msg.status_list[-1].status
        
        # 状态为3表示目标已到达
        if status == 3:  # SUCCEEDED
            rospy.loginfo(f"到达目标点 {self.current_goal_index + 1}")
            
            if self.mission_stage == "NAVIGATION":
                # 如果是最后一个目标点，等待稳定后开始数字识别任务
                if self.current_goal_index == len(self.goals) - 1:
                    # 设置标志表示已到达最终目标点
                    self.reached_final_goal = True
                    # 关键修复：添加延迟确保机器人完全稳定后再启动数字识别
                    rospy.loginfo("已到达河边，等待位置稳定...")
                    rospy.sleep(12)  # 等待5秒钟让机器人稳定
                    rospy.loginfo("现在开始识别数字...")
                    self.start_number_identification()
                else:
                    # 前往下一个目标点
                    self.current_goal_index += 1
                    # 先重新发布当前位姿减少漂移
                    self.publish_current_pose_as_initial()
                    rospy.sleep(5)  # 等待位姿稳定
                    self.publish_next_goal()
            
            elif self.mission_stage == "AFTER_BRIDGE":
                # 过桥后的目标点导航
                rospy.loginfo(f"过桥后到达目标点 {self.current_goal_index + 1}")
                
                # 启动数字识别节点来扫描此位置的数字
                rospy.loginfo("开始在当前位置扫描数字...")
                self.start_after_bridge_recognition()
                
                # 等待几秒钟以确保数字识别节点有充分时间运行
                rospy.sleep(10)
                
                # 如果已经是最后一个目标点，标记过桥后任务完成
                if self.current_goal_index == len(self.after_bridge_goals) - 1:
                    rospy.loginfo("已到达过桥后的最后一个目标点")
                    self.after_bridge_completed = True
                else:
                    # 停止当前数字识别节点
                    self.stop_after_bridge_recognition()
                    
                    # 前往下一个目标点
                    self.current_goal_index += 1
                    # 先重新发布当前位姿减少漂移
                    self.publish_current_pose_as_initial()
                    rospy.sleep(2)  # 等待位姿稳定
                    self.publish_next_goal()
    
    def number_callback(self, msg):
        """处理识别到的数字"""
        digit = msg.data
        rospy.loginfo(f"收到识别的数字: {digit}")
        
        if self.mission_stage == "NUMBER_IDENTIFICATION":
            # 将识别到的数字添加到列表中
            self.recognized_digits.append(digit)
            
            # 如果收集了足够多的数字样本，确定最少出现的数字
            if len(self.recognized_digits) >= 10:
                self.number_recognized = True
                
                # 统计每个数字出现的次数
                digit_counts = {}
                for d in self.recognized_digits:
                    if d in digit_counts:
                        digit_counts[d] += 1
                    else:
                        digit_counts[d] = 1
                
                # 找出出现次数最多的数字 (修改为最多而不是最少)
                most_common_digit = max(digit_counts.items(), key=lambda x: x[1])
                
                # 保存这个数字，在过桥后的任务中使用
                self.after_bridge_target_number = most_common_digit[0]
                
                rospy.loginfo(f"确定最常见的数字是: {self.after_bridge_target_number} (出现 {most_common_digit[1]} 次)")
                rospy.loginfo("数字识别阶段完成，准备过桥...")
                
                # 停止数字识别节点
                if self.number_process:
                    try:
                        self.number_process.terminate()
                        rospy.loginfo("数字识别节点已停止")
                    except:
                        rospy.logwarn("停止数字识别节点失败")
                
                # 开始过桥任务
                self.start_bridge_crossing()
    
    def start_number_identification(self):
        """启动数字识别节点"""
        self.mission_stage = "NUMBER_IDENTIFICATION"
        rospy.loginfo("启动数字识别节点...")
        
        try:
            # 启动数字识别节点
            self.number_process = subprocess.Popen(
                ["rosrun", "me5413_world", "number_identification.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rospy.loginfo("数字识别节点已启动")
        except Exception as e:
            rospy.logerr(f"启动数字识别节点失败: {e}")
            # 如果启动失败，直接进入过桥阶段
            self.start_bridge_crossing()
    
    def start_bridge_crossing(self):
        """启动过桥任务"""
        self.mission_stage = "BRIDGE_CROSSING"
        rospy.loginfo("启动橙色障碍桶检测节点...")
        
        try:
            # 启动橙色障碍桶检测节点
            self.auto_detector_process = subprocess.Popen(
                ["rosrun", "me5413_world", "auto_detector_controller.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rospy.loginfo("橙色障碍桶检测节点已启动")
            
            # 监听过桥完成的信号
            # 在实际应用中，这里应该添加一个订阅者来接收过桥完成的消息
            # 简化处理：假设20秒后过桥完成
            rospy.Timer(rospy.Duration(20), self.bridge_crossing_completed_callback, oneshot=True)
            
        except Exception as e:
            rospy.logerr(f"启动橙色障碍桶检测节点失败: {e}")
            rospy.signal_shutdown("任务执行失败")
    
    def bridge_crossing_completed_callback(self, event):
        """过桥完成后的回调"""
        rospy.loginfo("过桥任务完成，准备进入过桥后阶段...")
        
        # 停止过桥相关的节点
        if self.auto_detector_process:
            try:
                self.auto_detector_process.terminate()
                rospy.loginfo("橙色障碍桶检测节点已停止")
            except:
                rospy.logwarn("停止橙色障碍桶检测节点失败")
        
        # 标记过桥完成
        self.bridge_crossing_completed = True
        
        # 设置过桥后阶段
        self.start_after_bridge_phase()
    
    def start_after_bridge_phase(self):
        """开始过桥后的任务阶段"""
        self.mission_stage = "AFTER_BRIDGE"
        self.current_goal_index = 0  # 重置目标点索引
        
        rospy.loginfo(f"开始过桥后任务阶段，目标是找到数字: {self.after_bridge_target_number}")
        
        # 确保有目标数字
        if not self.after_bridge_target_number:
            self.after_bridge_target_number = "5"  # 默认值
            rospy.logwarn(f"没有识别到目标数字，使用默认值: {self.after_bridge_target_number}")
        
        # 开始导航到第一个过桥后的目标点
        self.publish_next_goal()
    
    def start_after_bridge_recognition(self):
        """启动过桥后的数字识别节点"""
        rospy.loginfo(f"启动过桥后数字识别节点，寻找数字: {self.after_bridge_target_number}")
        
        # 先发布目标数字
        self.recognized_number_pub.publish(self.after_bridge_target_number)
        rospy.sleep(1)  # 确保消息已发布
        
        try:
            # 启动数字识别节点
            self.after_bridge_process = subprocess.Popen(
                ["rosrun", "me5413_world", "after_bridge_front_recognize.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rospy.loginfo("过桥后数字识别节点已启动")
            
            # 设置超时机制
            rospy.Timer(rospy.Duration(30), self.after_bridge_recognition_timeout, oneshot=True)
            
        except Exception as e:
            rospy.logerr(f"启动过桥后数字识别节点失败: {e}")
            # 如果失败，继续下一个目标点
            if self.current_goal_index < len(self.after_bridge_goals) - 1:
                self.current_goal_index += 1
                self.publish_next_goal()
            else:
                self.after_bridge_completed = True
    
    def stop_after_bridge_recognition(self):
        """停止过桥后的数字识别节点"""
        if self.after_bridge_process:
            try:
                self.after_bridge_process.terminate()
                rospy.loginfo("过桥后数字识别节点已停止")
                self.after_bridge_process = None
            except:
                rospy.logwarn("停止过桥后数字识别节点失败")
    
    def after_bridge_recognition_timeout(self, event):
        """数字识别超时处理"""
        # 检查节点是否仍在运行
        if self.after_bridge_process and self.after_bridge_process.poll() is None:
            rospy.loginfo("在当前位置未找到目标数字，继续下一个位置")
            self.stop_after_bridge_recognition()
            
            # 如果不是最后一个目标点，继续下一个
            if self.current_goal_index < len(self.after_bridge_goals) - 1:
                self.current_goal_index += 1
                self.publish_next_goal()
            else:
                rospy.loginfo("已检查所有位置，未找到目标数字")
                self.after_bridge_completed = True
    
    def start_mission(self):
        """开始整个任务"""
        rospy.loginfo("开始导航任务...")
        self.publish_next_goal()
        
        # 进入主循环
        rate = rospy.Rate(1)  # 1Hz
        wait_counter = 0
        stabilization_time = 10  # 10秒稳定时间
        
        while not rospy.is_shutdown():
            # 如果到达了最终目标点但还没有标记导航完成，先等待稳定
            if self.reached_final_goal and not self.navigation_completed:
                wait_counter += 1
                rospy.loginfo(f"等待位置稳定 ({wait_counter}/{stabilization_time} 秒)...")
                
                if wait_counter >= stabilization_time:
                    rospy.loginfo("位置已充分稳定，导航阶段完成")
                    self.navigation_completed = True
                    wait_counter = 0
            
            # 如果导航已完成且当前仍在导航阶段，开始数字识别
            elif self.mission_stage == "NAVIGATION" and self.navigation_completed:
                rospy.loginfo("正式开始数字识别任务...")
                self.start_number_identification()
            
            # 如果数字识别已完成，开始过桥
            elif self.mission_stage == "NUMBER_IDENTIFICATION" and self.number_recognized:
                rospy.loginfo("数字识别任务完成，准备过桥...")
                self.start_bridge_crossing()
            
            # 如果过桥已完成，开始过桥后任务
            elif self.mission_stage == "BRIDGE_CROSSING" and self.bridge_crossing_completed:
                rospy.loginfo("过桥任务完成，开始过桥后任务...")
                self.start_after_bridge_phase()
            
            # 检查过桥后任务是否完成
            elif self.mission_stage == "AFTER_BRIDGE":
                # 检查after_bridge_process是否已终止 (可能找到了目标数字)
                if self.after_bridge_process and self.after_bridge_process.poll() is not None:
                    rospy.loginfo("过桥后数字识别节点已退出，可能已找到目标数字")
                    self.after_bridge_process = None
                    self.after_bridge_completed = True
                
                # 如果过桥后任务已完成，终止程序
                if self.after_bridge_completed:
                    rospy.loginfo("✅ 所有任务已完成，程序即将退出")
                    rospy.signal_shutdown("任务完成")
                    break
            
            rate.sleep()
    
    def shutdown(self):
        """清理进程并正常关闭"""
        rospy.loginfo("关闭所有进程...")
        
        # 关闭所有子进程
        for process in [self.number_process, self.auto_detector_process, self.after_bridge_process]:
            if process:
                try:
                    process.terminate()
                except:
                    pass
        
        rospy.loginfo("集成任务控制器已关闭")

if __name__ == '__main__':
    try:
        controller = IntegratedMissionController()
        rospy.on_shutdown(controller.shutdown)
        controller.start_mission()
    except rospy.ROSInterruptException:
        pass