#!/usr/bin/env python3
# coding=utf-8

import rospy
import cv2
import numpy as np
from pytesseract import pytesseract
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from collections import defaultdict

class NumberCountingNode:
    def __init__(self):
        rospy.init_node('number_counting_node')
        self.bridge = CvBridge()
        
        # 降低回调频率以减轻计算负担
        self.image_sub = rospy.Subscriber("/second/img_second", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.number_pub = rospy.Publisher("/recognized_number", String, queue_size=1)
        
        # 处理控制
        self.processing = False
        self.skip_count = 0
        self.process_every_n_frames = 10  # 每30帧处理一次
        
        # 历史记录，提高稳定性
        self.last_digits = []
        self.max_history = 5
        
        rospy.loginfo("数字计数节点已初始化")
        
    def image_callback(self, msg):
        # 跳过一些帧以减轻CPU负担
        self.skip_count += 1
        if self.skip_count % self.process_every_n_frames != 0:
            return
        
        if self.processing:
            return  # 如果上一帧还在处理，跳过此帧
        
        self.processing = True
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 顺时针旋转90度
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # 裁切图像，只保留中间3/5部分
            width = cv_image.shape[1]
            left_cut = width // 5  # 左侧1/5将被裁剪
            right_cut = width - (width // 5)  # 右侧1/5将被裁剪
            cv_image = cv_image[:, left_cut:right_cut]  # 保留中间3/5
            
            # 处理图像
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
        except Exception as e:
            rospy.logerr(f"处理图像时出错: {e}")
        finally:
            self.processing = False
    
    def non_max_suppression(self, boxes, overlapThresh=0.3):
        """
        应用非极大值抑制(NMS)算法过滤重叠的边界框
        
        参数:
            boxes: 格式为(x, y, w, h, digit, confidence)的候选框列表
            overlapThresh: 重叠阈值，超过这个阈值的框会被过滤掉
            
        返回:
            过滤后的边界框列表
        """
        # 如果没有框，直接返回空列表
        if len(boxes) == 0:
            return []
        
        # 初始化已选择的索引列表
        pick = []
        
        # 提取坐标
        x = np.array([box[0] for box in boxes])
        y = np.array([box[1] for box in boxes])
        w = np.array([box[2] for box in boxes])
        h = np.array([box[3] for box in boxes])
        
        # 计算每个框的面积和右下角坐标
        area = w * h
        xx2 = x + w
        yy2 = y + h
        
        # 按面积排序（从小到大）
        idxs = np.argsort(area)
        
        # 循环处理，直到没有边界框
        while len(idxs) > 0:
            # 取最后一个（最大面积）
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # 找出所有与当前框重叠的框
            xx1 = np.maximum(x[i], x[idxs[:last]])
            yy1 = np.maximum(y[i], y[idxs[:last]])
            xx2_overlap = np.minimum(xx2[i], xx2[idxs[:last]])
            yy2_overlap = np.minimum(yy2[i], yy2[idxs[:last]])
            
            # 计算重叠区域的宽度和高度
            w_overlap = np.maximum(0, xx2_overlap - xx1)
            h_overlap = np.maximum(0, yy2_overlap - yy1)
            
            # 计算重叠面积与较小框面积的比例
            overlap = (w_overlap * h_overlap) / area[idxs[:last]]
            
            # 删除重叠框
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
        # 返回保留的框
        return [boxes[i] for i in pick]
    
    def process_image(self, image_color):
        # 显示原始图像
        original_img = image_color.copy()
        
        # 转为灰度图
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_gray = clahe.apply(image_gray)
        
        # 高斯模糊去噪
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        # 自适应阈值处理
        thresh = cv2.adaptiveThreshold(
            image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 使用边缘检测增强数字轮廓
        edges = cv2.Canny(image_gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 候选框列表，格式为 (x, y, w, h, digit, confidence)
        digit_candidates = []
        
        for contour in contours:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            area = w * h
            
            # 过滤掉太小或太大的区域
            min_area = 100  # 最小面积
            max_area = image_color.shape[0] * image_color.shape[1] // 10  # 最大面积为图像的1/10
            
            if min_area < area < max_area and 0.2 < aspect_ratio < 2.0:
                # 提取感兴趣区域
                roi = image_gray[y:y+h, x:x+w]
                
                # 检查区域内像素密度
                pixel_density = np.sum(thresh[y:y+h, x:x+w] > 0) / (w * h)
                
                # 如果密度太低或太高，可能不是数字
                if 0.1 < pixel_density < 0.9:
                    # 调整ROI大小到合适的范围
                    target_height = 100
                    scale = target_height / h
                    target_width = int(w * scale)
                    roi_resized = cv2.resize(roi, (target_width, target_height))
                    
                    # 提高对比度
                    roi_enhanced = cv2.convertScaleAbs(roi_resized, alpha=1.5, beta=10)
                    
                    # 使用tesseract识别数字 - 只识别1-9
                    text = pytesseract.image_to_string(roi_enhanced, config='--oem 3 --psm 10 -c tessedit_char_whitelist=123456789')
                    text = ''.join(filter(lambda c: c in '123456789', text))
                    
                    if text and len(text) == 1:
                        # 添加置信度估计 - 使用像素密度作为简单的置信度指标
                        confidence = pixel_density
                        digit_candidates.append((x, y, w, h, text, confidence))
        
        # 应用非极大值抑制，过滤重叠框
        filtered_candidates = self.non_max_suppression(digit_candidates, overlapThresh=0.3)
        
        # 统计过滤后的数字出现次数
        digit_counts = defaultdict(int)
        digit_boxes = defaultdict(list)
        
        # 在图像上显示过滤后的候选框
        for x, y, w, h, digit, confidence in filtered_candidates:
            digit_counts[digit] += 1
            digit_boxes[digit].append((x, y, w, h))
            
            # 在图像上标记识别到的数字
            cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(original_img, f"{digit}", 
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # 选择出现次数最少的数字
        recognized_digit = None
        if digit_counts:
            # 找出出现次数最少的数字
            min_digit = min(digit_counts.items(), key=lambda x: x[1])
            digit = min_digit[0]
            count = min_digit[1]
            
            recognized_digit = digit
            
            # 在图像上标记出现次数最少的数字
            if digit in digit_boxes and digit_boxes[digit]:
                # 获取第一个边界框
                x, y, w, h = digit_boxes[digit][0]
                
                # 标记出现次数最少的数字
                cv2.rectangle(original_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(original_img, f"Least Common: {digit} (count: {count})", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 添加到历史记录
            self.last_digits.append(recognized_digit)
            if len(self.last_digits) > self.max_history:
                self.last_digits.pop(0)
            
            # 使用历史记录中出现最多的数字作为最终识别结果，增加稳定性
            history_counts = defaultdict(int)
            for d in self.last_digits:
                history_counts[d] += 1
            
            final_digit = max(history_counts.items(), key=lambda x: x[1])[0]
            
            rospy.loginfo(f"当前帧中出现次数最少的数字: {recognized_digit} (次数: {count})")
            rospy.loginfo(f"历史稳定后的数字: {final_digit} (历史记录: {self.last_digits})")
            
            # 发布识别结果
            self.number_pub.publish(recognized_digit)
        
        # 显示数字计数结果
        count_text = ", ".join([f"{d}: {c}" for d, c in sorted(digit_counts.items())])
        cv2.putText(original_img, f"Counts: {count_text}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示识别结果
        cv2.imshow("Number Counting", original_img)
        cv2.imshow("Preprocessed", thresh)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = NumberCountingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()