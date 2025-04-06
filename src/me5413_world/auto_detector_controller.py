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
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create motion control publisher
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        
        # Create bridge control publisher
        self.bridge_pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=1)
        
        # Create image subscription
        self.rgb_sub = rospy.Subscriber("/front/image_raw", Image, self.camera_callback, queue_size=10)
        
        # Control parameters
        self.linear_speed = 0.3       # Forward speed
        self.angular_speed = 0.3      # Turning speed 0.5
        self.close_speed = 0.2        # Approach speed
        self.stop_distance = 100000   # Stop distance (area threshold)
        self.screen_coverage = 0.7    # Screen coverage threshold, indicating how much of the screen the orange bucket should cover
        
        # Final approach flags
        self.final_approach = False
        self.final_approach_count = 0
        self.final_approach_time = 3.0  # Final approach time in seconds
        self.final_approach_start_time = None
        
        # Bridge crossing flags
        self.crossing_bridge = False
        self.bridge_opened = False
        self.cross_bridge_start_time = None
        self.cross_bridge_time = 7.0  # Bridge crossing time, program ends after 5 seconds
        
        # Control stability parameters
        self.last_action = None
        self.action_count = 0
        self.max_action_repeat = 3  # Maximum number of consecutive identical actions
        
        # Display settings
        self.show_image = True
        
        # Status flags
        self.mission_complete = False
        
        rospy.loginfo("Orange bucket auto controller initialized")
    
    def open_bridge(self):
        """Open the bridge barrier"""
        if not self.bridge_opened:
            rospy.loginfo("🌉 Opening bridge barrier...")
            bridge_msg = Bool()
            bridge_msg.data = True
            self.bridge_pub.publish(bridge_msg)
            self.bridge_opened = True
    
    def analyze_orange_bucket(self, image):
        """Analyze the orange bucket in the image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_area = image.shape[0] * image.shape[1]  # Total image area

        # Orange HSV range
        lower_orange = np.array([10, 80, 110])
        upper_orange = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Morphological processing
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, mask

        # Filter small contours
        filtered = [c for c in contours if cv2.contourArea(c) > 300]
        if len(filtered) == 0:
            return None, mask

        # Merge all contours and calculate bounding box
        all_pts = np.vstack(filtered)
        x, y, w, h = cv2.boundingRect(all_pts)
        area = w * h
        cx = x + w // 2
        img_width = image.shape[1]
        center_offset = cx - img_width // 2
        wh_ratio = w / float(h)
        
        # Calculate coverage ratio
        coverage_ratio = area / image_area

        # Return information dictionary
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
        """Decide vehicle action based on bucket information"""
        # Check if crossing the bridge, if so continue the bridge crossing action
        if self.crossing_bridge:
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - self.cross_bridge_start_time
            
            if elapsed_time > self.cross_bridge_time:
                rospy.loginfo("🏁 Bridge crossing complete! Program will exit in 2 seconds...")
                # Stop the vehicle
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                rospy.sleep(2)  # Wait for 2 seconds
                rospy.signal_shutdown("Bridge crossing task completed")
                sys.exit(0)  # Ensure complete exit
            else:
                return "CROSS_BRIDGE", f"🌉 Crossing bridge at full speed... ({int(elapsed_time)}/{int(self.cross_bridge_time)} sec)"
        
        if info is None:
            self.final_approach = False
            return "SEARCH", "❌ Orange bucket not detected, searching"

        if self.final_approach:
            # If in final approach stage
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - self.final_approach_start_time
            
            if elapsed_time > self.final_approach_time:
                # Open bridge barrier
                self.open_bridge()
                # Enter bridge crossing stage
                self.crossing_bridge = True
                self.cross_bridge_start_time = rospy.Time.now().to_sec()
                return "CROSS_BRIDGE", "🌉 Barrier opened, starting full speed bridge crossing!"
            else:
                return "FINAL_APPROACH", f"🚀 Final approach in progress... ({int(elapsed_time)}/{int(self.final_approach_time)} sec)"

        # Check if area exceeds threshold, directly enter bridge crossing mode
        if info["area"] > self.stop_distance:
            self.final_approach = True
            self.final_approach_start_time = rospy.Time.now().to_sec()
            return "FINAL_APPROACH", "🚀 Orange bucket has reached specified size, preparing to cross bridge"

        # Normal detection logic
        if info["full_screen"]:
            self.final_approach = True
            self.final_approach_start_time = rospy.Time.now().to_sec()
            return "FINAL_APPROACH", "🚀 Bucket covers most of the screen, starting final approach"

        if not info["is_well_formed"]:
            # Different adjustment strategies based on bucket position
            if info["center_offset"] > 0:
                return "ADJUST_RIGHT", "🔄 Bucket shape abnormal and right-biased, special adjustment"
            else:
                return "ADJUST_LEFT", "🔄 Bucket shape abnormal and left-biased, special adjustment"

        if not info["is_centered"]:
            if info["center_offset"] > 0:
                return "RIGHT", "➡️ Bucket right-biased, turning right"
            else:
                return "LEFT", "⬅️ Bucket left-biased, turning left"

        if info["is_too_large"]:
            return "SLOW_APPROACH", "🐢 Orange bucket is close, slow approach"

        if not info["is_large"]:
            return "FORWARD", "⬆️ Bucket centered but far, moving forward"

        return "APPROACH", "✅ Bucket found, approaching"

    def execute_action(self, action_code):
        """Execute action"""
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
            # Posture adjustment - small rotation in place
            twist.linear.x = -0.01
            twist.angular.z = 0.2
        elif action_code == "ADJUST_LEFT":
            # Adjustment strategy for abnormal shape and left-biased bucket
            twist.linear.x = -0.15  # Slight backward
            twist.angular.z = -0.3   # Larger left turn angle
        elif action_code == "ADJUST_RIGHT":
            # Adjustment strategy for abnormal shape and right-biased bucket
            twist.linear.x = 0.15  # Slight forward
            twist.angular.z = 0.3  # Larger left turn angle
        elif action_code == "SEARCH":
            # Search mode - slow rotation
            twist.linear.x = 0.0
            twist.angular.z = 0.3
        elif action_code == "SLOW_APPROACH":
            # Slow approach
            twist.linear.x = 0.15
            twist.angular.z = 0.0
        elif action_code == "APPROACH":
            # Normal speed approach
            twist.linear.x = 0.25
            twist.angular.z = 0.0
        elif action_code == "FINAL_APPROACH":
            # Final approach - continue forward until covering the entire screen
            twist.linear.x = self.close_speed
            twist.angular.z = 0.0
        elif action_code == "CROSS_BRIDGE":
            # Full speed bridge crossing
            twist.linear.x = 0.5  # Higher speed for crossing bridge
            twist.angular.z = 0.0
        elif action_code == "MISSION_COMPLETE":
            # Mission complete - stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # Only show mission complete notification once
            if not self.mission_complete:
                rospy.loginfo("🏆 Mission complete! Orange bucket fully covers the screen")
                self.mission_complete = True
        
        # Publish velocity command
        self.cmd_vel_pub.publish(twist)
        
        # Record last executed action
        if self.last_action == action_code:
            self.action_count += 1
        else:
            self.last_action = action_code
            self.action_count = 1
            
        # If the same action is executed too many times, reduce control magnitude to avoid overshooting
        if self.action_count > self.max_action_repeat and action_code not in ["FINAL_APPROACH", "CROSS_BRIDGE"]:
            reduced_twist = Twist()
            reduced_twist.linear.x = twist.linear.x * 0.7
            reduced_twist.angular.z = twist.angular.z * 0.7
            self.cmd_vel_pub.publish(reduced_twist)
    
    def camera_callback(self, msg):
        """Camera callback function, process image and control motion"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("Format conversion error: %s", e)
            return

        # Analyze orange bucket in the image
        info, mask = self.analyze_orange_bucket(cv_image)
        
        # Decide next action
        action_code, action_desc = self.decide_action(info)
        
        # Execute action
        self.execute_action(action_code)
        
        # Display control information
        rospy.loginfo("🚗 Control: %s", action_desc)

        # Visualization
        if self.show_image and info is not None:
            x, y, w, h = info["bbox"]
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display action on image
            cv2.putText(cv_image, action_desc, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Display area and coverage on image
            cv2.putText(cv_image, f"Area: {info['area']}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(cv_image, f"Coverage: {info['coverage_ratio']:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Display offset
            cv2.putText(cv_image, f"Offset: {info['center_offset']}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.show_image:
            cv2.imshow("Camera View", cv_image)
            cv2.imshow("Orange Mask", mask)
            cv2.waitKey(1)

    def run(self):
        """Run controller"""
        rate = rospy.Rate(10)  # 10Hz
        try:
            while not rospy.is_shutdown():
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("User interrupt, stopping controller")
        finally:
            # Send stop command
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        controller = OrangeDetectorController()
        controller.run()
    except rospy.ROSInterruptException:
        pass