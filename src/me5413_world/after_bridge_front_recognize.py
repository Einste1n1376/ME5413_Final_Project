#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from pytesseract import pytesseract
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

class SimplifiedNumberDetector:
    def __init__(self):
        rospy.init_node('simplified_number_detector', anonymous=True)
        
        # 初始化工具
        self.bridge = CvBridge()
        
        # 初始化发布者
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # 初始化订阅者
        self.reference_number_sub = rospy.Subscriber('/recognized_number', String, self.reference_number_callback)
        self.camera_sub = rospy.Subscriber('/front/image_raw', Image, self.camera_callback, queue_size=1)
        
        # 状态变量
        self.reference_number = None
        self.last_recognized_digit = None
        self.processing_image = False
        self.match_found = False
        self.approach_start_time = None
        self.recognition_history = []  # 存储最近的识别结果
        self.history_max_size = 5      # 历史记录最大长度
        
        # 可视化设置 - 简化，只保留必要的可视化
        self.debug_images = True
        self.recognition_count = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        
        # 运动控制变量 - 简化靠近逻辑
        self.current_cmd = Twist()  # 当前的速度命令
        self.approach_speed = 0.5     # 降低速度为0.2m/s
        self.approach_duration = 3.0  # 只前进2秒
        
        # 创建一个定时器，持续发送速度命令
        self.cmd_timer = rospy.Timer(rospy.Duration(0.05), self.cmd_timer_callback)  # 20Hz更新速率
        
        # 验证ROS发布者是否正常工作
        rospy.sleep(1)
        test_twist = Twist()
        test_twist.linear.x = 0.0
        test_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(test_twist)
        rospy.loginfo("简化版数字识别器初始化完成")

    def cmd_timer_callback(self, event):
        """定时器回调函数，持续发送当前速度命令"""
        if self.match_found:
            self.cmd_vel_pub.publish(self.current_cmd)
            
    def reference_number_callback(self, msg):
        """存储参考数字"""
        self.reference_number = msg.data
        rospy.loginfo(f"接收到参考数字: {self.reference_number}")
    
    def camera_callback(self, msg):
        """处理摄像头图像，识别数字"""
        if self.processing_image or self.match_found:
            return
            
        self.processing_image = True
        try:
            # 转换ROS图像到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 识别图像中的数字
            recognized_digit, bounding_box = self.process_image(cv_image)
            
            # 处理识别结果
            if recognized_digit:
                self.recognition_count += 1
                self.last_recognized_digit = recognized_digit
                
                # 添加到历史记录
                self.recognition_history.append(recognized_digit)
                if len(self.recognition_history) > self.history_max_size:
                    self.recognition_history.pop(0)
                
                # 计算历史记录中出现最多的数字
                if len(self.recognition_history) >= 3:
                    digit_counts = {}
                    for digit in self.recognition_history:
                        if digit in digit_counts:
                            digit_counts[digit] += 1
                        else:
                            digit_counts[digit] = 1
                    
                    most_common_digit = max(digit_counts.items(), key=lambda x: x[1])
                    stable_digit = most_common_digit[0]
                    stable_count = most_common_digit[1]
                    
                    # 如果某个数字连续出现超过一半的历史记录，认为它是稳定的
                    if stable_count >= len(self.recognition_history) // 2:
                        rospy.loginfo(f"稳定识别到数字: {stable_digit} (出现频率: {stable_count}/{len(self.recognition_history)})")
                        
                        # 对比识别的数字与参考数字
                        if self.reference_number and stable_digit == self.reference_number:
                            rospy.loginfo(f"🎯 匹配成功! 稳定识别的数字 {stable_digit} 与参考数字相符")
                            self.successful_recognitions += 1
                            self.match_found = True
                            self.approach_start_time = rospy.Time.now().to_sec()
                            self.approach_target()
                        else:
                            if self.reference_number:
                                rospy.loginfo(f"❌ 匹配失败! 识别到 {stable_digit}，但参考数字是 {self.reference_number}")
                                self.failed_recognitions += 1
                
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误: {e}")
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
        finally:
            self.processing_image = False
    
    def find_largest_contour(self, binary_image):
        """查找图像中最大的轮廓"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # 找出面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 只处理面积足够大的轮廓 (过滤噪点)
        min_area = 500  # 最小面积阈值
        if cv2.contourArea(largest_contour) < min_area:
            return None, None
            
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 过滤不合理比例的边界框
        aspect_ratio = float(w) / h
        if not (0.2 < aspect_ratio < 2.0):  # 过滤太细长或太扁平的框
            return None, None
            
        return largest_contour, (x, y, w, h)
    
    def process_image(self, image_color):
        """处理图像，识别数字 - 简化版，只识别最大的数字区域"""
        # 创建可视化图像副本
        visual_img = image_color.copy()
        
        # 转为灰度图
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_gray = clahe.apply(image_gray)
        
        # 高斯模糊去噪
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        # 利用二值化反转图像，使数字为白色，背景为黑色
        _, binary_inv = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作改善数字形状
        kernel = np.ones((5, 5), np.uint8)
        morph_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 找到最大的轮廓
        largest_contour, bounding_box = self.find_largest_contour(morph_inv)
        
        recognized_digit = None
        
        if bounding_box is not None:
            x, y, w, h = bounding_box
            
            # 在图像上绘制边界框
            cv2.rectangle(visual_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 扩大ROI区域，确保数字完全包含在内
            padding = 10
            roi_x = max(0, x - padding)
            roi_y = max(0, y - padding)
            roi_w = min(w + 2*padding, image_gray.shape[1] - roi_x)
            roi_h = min(h + 2*padding, image_gray.shape[0] - roi_y)
            
            # 提取感兴趣区域
            roi = morph_inv[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # 调整ROI大小到标准尺寸以提高识别率
            roi_resized = cv2.resize(roi, (100, 100))
            
            # 配置tesseract只识别单个数字1-9
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
            
            # 尝试识别
            result = pytesseract.image_to_string(roi_resized, config=custom_config)
            chars = ''.join(filter(lambda c: c in '123456789', result))
            
            # 如果识别出数字，取第一个
            if chars:
                recognized_digit = chars[0]
                rospy.loginfo(f"识别到数字: {recognized_digit}")
                
                # 在图像上的边界框旁显示识别结果
                cv2.putText(visual_img, f"{recognized_digit}", 
                        (x+w+10, y+h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # 在图像上也显示识别结果
                cv2.putText(visual_img, f"识别结果: {recognized_digit}", 
                        (visual_img.shape[1]//2 - 150, visual_img.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # 显示状态信息
        cv2.putText(visual_img, f"参考数字: {self.reference_number}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 匹配状态
        if self.match_found:
            status_text = "状态: 匹配成功，正在靠近"
            status_color = (0, 255, 0)
        else:
            status_text = "状态: 识别中..."
            status_color = (0, 165, 255)
        
        cv2.putText(visual_img, status_text, 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 简化显示，只显示识别结果
        if self.debug_images:
            cv2.imshow("Number Recognition", visual_img)
            cv2.waitKey(1)
        
        return recognized_digit, bounding_box
    
    def approach_target(self):
        """向前靠近目标 - 简化版，只前进固定时间"""
        rospy.loginfo("🚗 向前靠近目标...")
        
        # 设置接近开始时间
        self.approach_start_time = rospy.Time.now().to_sec()
        
        # 重置任何先前的移动命令
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd) 
        rospy.sleep(0.2)  # 短暂停顿以重置移动状态
        
        # 初始化当前速度命令
        self.current_cmd = Twist()
        self.current_cmd.linear.x = self.approach_speed
        self.current_cmd.angular.z = 0.0
        
        # 直接发送第一个移动命令
        for _ in range(3):  # 多次发送确保被接收
            self.cmd_vel_pub.publish(self.current_cmd)
            rospy.sleep(0.05)
        
        rospy.loginfo(f"🚀 开始以 {self.approach_speed} m/s 的速度向前移动，将持续 {self.approach_duration} 秒")
    
    def run(self):
        """运行主循环"""
        rospy.loginfo("开始数字识别任务...")
        
        # 等待接收参考数字
        timeout = rospy.Time.now() + rospy.Duration(2)  # 10秒超时
        while self.reference_number is None and not rospy.is_shutdown():
            if rospy.Time.now() > timeout:
                rospy.logwarn("等待参考数字超时，使用默认值'4'")
                self.reference_number = "4"
                break
            rospy.loginfo("等待接收参考数字...")
            rospy.sleep(1)
        
        # 主循环
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            # 如果找到了匹配的数字，等待靠近过程完成
            if self.match_found:
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - self.approach_start_time
                
                # 简化版: 只检查是否达到预定时间
                if elapsed_time > self.approach_duration:
                    # 时间到，停止小车
                    self.current_cmd = Twist()  # 重置为零速度
                    for _ in range(3):  # 多次发送确保停止
                        self.cmd_vel_pub.publish(self.current_cmd)
                        rospy.sleep(0.1)
                    
                    rospy.loginfo("✅ 前进完成! 已找到匹配数字并靠近")
                    rospy.signal_shutdown("任务完成")
                    break
                
                # 简单记录当前状态
                if int(elapsed_time * 2) % 2 == 0:  # 每0.5秒记录一次
                    progress = int((elapsed_time / self.approach_duration) * 100)
                    rospy.loginfo(f"前进进度: {progress}%")
            
            rate.sleep()
    
    def shutdown(self):
        """清理资源"""
        # 停止定时器
        if hasattr(self, 'cmd_timer'):
            self.cmd_timer.shutdown()
        
        # 发送停止命令
        stop_cmd = Twist()
        for _ in range(3):  # 多次发送停止命令确保停止
            self.cmd_vel_pub.publish(stop_cmd)
            rospy.sleep(0.1)
        
        cv2.destroyAllWindows()
        rospy.loginfo("数字识别器已关闭")

if __name__ == '__main__':
    try:
        detector = SimplifiedNumberDetector()
        rospy.on_shutdown(detector.shutdown)
        detector.run()
    except rospy.ROSInterruptException:
        pass