#!/usr/bin/env python3
# coding=utf-8

import rospy
import cv2
import numpy as np
import subprocess
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import time
import sys

class OrangeDetectorController:
    def __init__(self):
        rospy.init_node("orange_detector_controller")
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 创建运动控制发布器
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        
        # 创建桥梁控制发布器
        self.bridge_pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=1)
        
        # 创建图像订阅
        self.rgb_sub = rospy.Subscriber("/front/image_raw", Image, self.camera_callback, queue_size=10)
        
        # 控制参数
        self.linear_speed = 0.3       # 前进速度
        self.angular_speed = 0.3      # 转向速度0.5
        self.close_speed = 0.2        # 贴近速度
        self.stop_distance = 100000   # 停止距离(区域阈值)
        self.screen_coverage = 0.7    # 屏幕覆盖率阈值，表示橙桶应覆盖画面的比例
        
        # 最接近阶段标志
        self.final_approach = False
        self.final_approach_count = 0
        self.final_approach_time = 3.0  # 最终接近时间，单位秒
        self.final_approach_start_time = None
        
        # 过桥阶段标志
        self.crossing_bridge = False
        self.bridge_opened = False
        self.cross_bridge_start_time = None
        self.cross_bridge_time = 7.0  # 过桥时间，5秒后结束程序
        
        # 控制稳定性参数
        self.last_action = None
        self.action_count = 0
        self.max_action_repeat = 3  # 连续执行相同动作的最大次数
        
        # 显示设置
        self.show_image = True
        
        # 状态标志
        self.mission_complete = False
        
        rospy.loginfo("橙色桶自动控制器已初始化")
    
    def open_bridge(self):
        """打开桥梁路障"""
        if not self.bridge_opened:
            rospy.loginfo("🌉 打开桥梁路障...")
            bridge_msg = Bool()
            bridge_msg.data = True
            self.bridge_pub.publish(bridge_msg)
            self.bridge_opened = True
    
    def analyze_orange_bucket(self, image):
        """分析图像中的橙色桶"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_area = image.shape[0] * image.shape[1]  # 总图像面积

        # 橙色HSV范围
        lower_orange = np.array([10, 80, 110])
        upper_orange = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # 形态学处理
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, mask

        # 过滤小轮廓
        filtered = [c for c in contours if cv2.contourArea(c) > 300]
        if len(filtered) == 0:
            return None, mask

        # 合并所有轮廓并计算边界框
        all_pts = np.vstack(filtered)
        x, y, w, h = cv2.boundingRect(all_pts)
        area = w * h
        cx = x + w // 2
        img_width = image.shape[1]
        center_offset = cx - img_width // 2
        wh_ratio = w / float(h)
        
        # 计算覆盖率
        coverage_ratio = area / image_area

        # 返回信息字典
        info = {
            "area": area,
            "center_offset": center_offset,
            "bbox": (x, y, w, h),
            "is_centered": abs(center_offset) < 50,
            "is_large": area > 10000,
            "is_too_large": area > self.stop_distance,
            "is_well_formed": 0.4 < wh_ratio < 1,
            "coverage_ratio": coverage_ratio,
            "full_screen": coverage_ratio > self.screen_coverage
        }

        return info, mask
    
    def decide_action(self, info):
        """根据桶的信息决定车辆动作"""
        # 检查是否正在过桥，如果是则继续过桥动作
        if self.crossing_bridge:
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - self.cross_bridge_start_time
            
            if elapsed_time > self.cross_bridge_time:
                rospy.loginfo("🏁 过桥完成！程序将在2秒后退出...")
                # 停止小车
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(2)  # 等待2秒
                rospy.signal_shutdown("过桥任务完成")
                sys.exit(0)  # 确保完全退出
            else:
                return "CROSS_BRIDGE", f"🌉 全速过桥中... ({int(elapsed_time)}/{int(self.cross_bridge_time)}秒)"
        
        if info is None:
            self.final_approach = False
            return "SEARCH", "❌ 未检测到橙色桶，搜索中"

        if self.final_approach:
            # 如果已进入最终接近阶段
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - self.final_approach_start_time
            
            if elapsed_time > self.final_approach_time:
                # 打开桥梁路障
                self.open_bridge()
                # 进入过桥阶段
                self.crossing_bridge = True
                self.cross_bridge_start_time = rospy.Time.now().to_sec()
                return "CROSS_BRIDGE", "🌉 路障已开启，开始全速过桥！"
            else:
                return "FINAL_APPROACH", f"🚀 最终接近中... ({int(elapsed_time)}/{int(self.final_approach_time)}秒)"

        # 检查区域是否超过阈值，直接进入过桥模式
        if info["area"] > self.stop_distance:
            self.final_approach = True
            self.final_approach_start_time = rospy.Time.now().to_sec()
            return "FINAL_APPROACH", "🚀 橙桶已到达指定大小，准备过桥"

        # 正常检测逻辑
        if info["full_screen"]:
            self.final_approach = True
            self.final_approach_start_time = rospy.Time.now().to_sec()
            return "FINAL_APPROACH", "🚀 桶已覆盖大部分屏幕，开始最终接近"

        if not info["is_well_formed"]:
            # 根据桶的位置决定不同的调整策略
            if info["center_offset"] > 0:
                return "ADJUST_RIGHT", "🔄 桶形状异常且偏右，特殊调整中"
            else:
                return "ADJUST_LEFT", "🔄 桶形状异常且偏左，特殊调整中"

        if not info["is_centered"]:
            if info["center_offset"] > 0:
                return "RIGHT", "➡️ 桶偏右，右转"
            else:
                return "LEFT", "⬅️ 桶偏左，左转"

        if info["is_too_large"]:
            return "SLOW_APPROACH", "🐢 橙桶很近，缓慢接近"

        if not info["is_large"]:
            return "FORWARD", "⬆️ 桶居中但远，前进靠近"

        return "APPROACH", "✅ 已找到桶，接近中"

    def execute_action(self, action_code):
        """执行动作"""
        twist = Twist()
        
        if action_code == "FORWARD":
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.1
        elif action_code == "LEFT":
            twist.linear.x = 0.1
            twist.angular.z = self.angular_speed
        elif action_code == "RIGHT":
            twist.linear.x = 0.1
            twist.angular.z = -self.angular_speed
        elif action_code == "STOP":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif action_code == "ADJUST":
            # 微调姿态 - 原地小幅旋转
            twist.linear.x = -0.01
            twist.angular.z = 0.2
        elif action_code == "ADJUST_LEFT":
            # 桶形状异常且偏左的调整策略
            twist.linear.x = -0.15  # 轻微后退
            twist.angular.z = -0.3   # 较大左转角度
        elif action_code == "ADJUST_RIGHT":
            # 桶形状异常且偏右的调整策略
            twist.linear.x = 0.15  # 轻微前进
            twist.angular.z = 0.3  # 较大左转角度
        elif action_code == "SEARCH":
            # 搜索模式 - 缓慢转圈
            twist.linear.x = 0.0
            twist.angular.z = 0.3
        elif action_code == "SLOW_APPROACH":
            # 缓慢接近
            twist.linear.x = 0.15
            twist.angular.z = 0.0
        elif action_code == "APPROACH":
            # 正常速度接近
            twist.linear.x = 0.25
            twist.angular.z = 0.0
        elif action_code == "FINAL_APPROACH":
            # 最终接近 - 继续前进直到覆盖整个屏幕
            twist.linear.x = self.close_speed
            twist.angular.z = 0.0
        elif action_code == "CROSS_BRIDGE":
            # 全速过桥
            twist.linear.x = 0.5  # 使用更高速度过桥
            twist.angular.z = 0.0
        elif action_code == "MISSION_COMPLETE":
            # 任务完成 - 停止
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # 只在第一次显示任务完成通知
            if not self.mission_complete:
                rospy.loginfo("🏆 任务完成！橙色桶已完全覆盖屏幕")
                self.mission_complete = True
        
        # 发布速度命令
        self.cmd_vel_pub.publish(twist)
        
        # 记录最后执行的动作
        if self.last_action == action_code:
            self.action_count += 1
        else:
            self.last_action = action_code
            self.action_count = 1
            
        # 如果同一动作执行次数过多，则减小控制量避免过冲
        if self.action_count > self.max_action_repeat and action_code not in ["FINAL_APPROACH", "CROSS_BRIDGE"]:
            reduced_twist = Twist()
            reduced_twist.linear.x = twist.linear.x * 0.7
            reduced_twist.angular.z = twist.angular.z * 0.7
            self.cmd_vel_pub.publish(reduced_twist)
    
    def camera_callback(self, msg):
        """相机回调函数，处理图像并控制运动"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("格式转换错误: %s", e)
            return

        # 分析图像中的橙色桶
        info, mask = self.analyze_orange_bucket(cv_image)
        
        # 决定下一步动作
        action_code, action_desc = self.decide_action(info)
        
        # 执行动作
        self.execute_action(action_code)
        
        # 显示控制信息
        rospy.loginfo("🚗 控制: %s", action_desc)

        # 可视化
        if self.show_image and info is not None:
            x, y, w, h = info["bbox"]
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 在图像上显示动作
            cv2.putText(cv_image, action_desc, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 在图像上显示面积和覆盖率
            cv2.putText(cv_image, f"Area: {info['area']}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(cv_image, f"Coverage: {info['coverage_ratio']:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 显示偏移量
            cv2.putText(cv_image, f"Offset: {info['center_offset']}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.show_image:
            cv2.imshow("Camera View", cv_image)
            cv2.imshow("Orange Mask", mask)
            cv2.waitKey(1)

    def run(self):
        """运行控制器"""
        rate = rospy.Rate(10)  # 10Hz
        try:
            while not rospy.is_shutdown():
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("用户中断，停止控制器")
        finally:
            # 发送停止命令
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        controller = OrangeDetectorController()
        controller.run()
    except rospy.ROSInterruptException:
        pass