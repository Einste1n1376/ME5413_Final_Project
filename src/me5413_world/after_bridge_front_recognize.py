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
        
        # Initialize tools
        self.bridge = CvBridge()
        
        # Initialize publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Initialize subscribers
        self.reference_number_sub = rospy.Subscriber('/recognized_number', String, self.reference_number_callback)
        self.camera_sub = rospy.Subscriber('/front/image_raw', Image, self.camera_callback, queue_size=1)
        
        # Status variables
        self.reference_number = None
        self.last_recognized_digit = None
        self.processing_image = False
        self.match_found = False
        self.approach_start_time = None
        self.recognition_history = []  # Store recent recognition results
        self.history_max_size = 5      # Maximum history length
        
        # Visualization settings - simplified, only keep necessary visualization
        self.debug_images = True
        self.recognition_count = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        
        # Motion control variables - simplified approach logic
        self.current_cmd = Twist()  # Current velocity command
        self.approach_speed = 0.5     # Reduced speed to 0.5m/s
        self.approach_duration = 3.0  # Only move forward for 3 seconds
        
        # Create a timer to continuously send velocity commands
        self.cmd_timer = rospy.Timer(rospy.Duration(0.05), self.cmd_timer_callback)  # 20Hz update rate
        
        # Verify ROS publisher is working properly
        rospy.sleep(1)
        test_twist = Twist()
        test_twist.linear.x = 0.0
        test_twist.angular.z = 0.0
        self.cmd_vel_pub.publish(test_twist)
        rospy.loginfo("Simplified number detector initialized")

    def cmd_timer_callback(self, event):
        """Timer callback function, continuously send current velocity command"""
        if self.match_found:
            self.cmd_vel_pub.publish(self.current_cmd)
            
    def reference_number_callback(self, msg):
        """Store reference number"""
        self.reference_number = msg.data
        rospy.loginfo(f"Received reference number: {self.reference_number}")
    
    def camera_callback(self, msg):
        """Process camera image, recognize numbers"""
        if self.processing_image or self.match_found:
            return
            
        self.processing_image = True
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Recognize numbers in the image
            recognized_digit, bounding_box = self.process_image(cv_image)
            
            # Process recognition results
            if recognized_digit:
                self.recognition_count += 1
                self.last_recognized_digit = recognized_digit
                
                # Add to history
                self.recognition_history.append(recognized_digit)
                if len(self.recognition_history) > self.history_max_size:
                    self.recognition_history.pop(0)
                
                # Calculate the most frequent number in history
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
                    
                    # If a number appears more than half of the history, consider it stable
                    if stable_count >= len(self.recognition_history) // 2:
                        rospy.loginfo(f"Stable number detected: {stable_digit} (Frequency: {stable_count}/{len(self.recognition_history)})")
                        
                        # Compare recognized number with reference number
                        if self.reference_number and stable_digit == self.reference_number:
                            rospy.loginfo(f"🎯 Match success! Stable recognized number {stable_digit} matches reference number")
                            self.successful_recognitions += 1
                            self.match_found = True
                            self.approach_start_time = rospy.Time.now().to_sec()
                            self.approach_target()
                        else:
                            if self.reference_number:
                                rospy.loginfo(f"❌ Match failed! Detected {stable_digit}, but reference number is {self.reference_number}")
                                self.failed_recognitions += 1
                
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
        finally:
            self.processing_image = False
    
    def find_largest_contour(self, binary_image):
        """Find the largest contour in the image"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Only process contours with sufficient area (filter noise)
        min_area = 500  # Minimum area threshold
        if cv2.contourArea(largest_contour) < min_area:
            return None, None
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter bounding boxes with unreasonable proportions
        aspect_ratio = float(w) / h
        if not (0.2 < aspect_ratio < 2.0):  # Filter too narrow or too flat boxes
            return None, None
            
        return largest_contour, (x, y, w, h)
    
    def process_image(self, image_color):
        """Process image, recognize digits - simplified version, only recognizes the largest digit area"""
        # Create visualization image copy
        visual_img = image_color.copy()
        
        # Convert to grayscale
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_gray = clahe.apply(image_gray)
        
        # Gaussian blur for noise reduction
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        # Use binary thresholding to invert the image, making digits white and background black
        _, binary_inv = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to improve digit shape
        kernel = np.ones((5, 5), np.uint8)
        morph_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find the largest contour
        largest_contour, bounding_box = self.find_largest_contour(morph_inv)
        
        recognized_digit = None
        
        if bounding_box is not None:
            x, y, w, h = bounding_box
            
            # Draw bounding box on the image
            cv2.rectangle(visual_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Enlarge ROI area to ensure the digit is fully contained
            padding = 10
            roi_x = max(0, x - padding)
            roi_y = max(0, y - padding)
            roi_w = min(w + 2*padding, image_gray.shape[1] - roi_x)
            roi_h = min(h + 2*padding, image_gray.shape[0] - roi_y)
            
            # Extract region of interest
            roi = morph_inv[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Resize ROI to standard size to improve recognition rate
            roi_resized = cv2.resize(roi, (100, 100))
            
            # Configure tesseract to only recognize single digits 1-9
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
            
            # Attempt recognition
            result = pytesseract.image_to_string(roi_resized, config=custom_config)
            chars = ''.join(filter(lambda c: c in '123456789', result))
            
            # If a digit is recognized, take the first one
            if chars:
                recognized_digit = chars[0]
                rospy.loginfo(f"Digit recognized: {recognized_digit}")
                
                # Display recognition result next to the bounding box
                cv2.putText(visual_img, f"{recognized_digit}", 
                        (x+w+10, y+h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Also display recognition result on the image
                cv2.putText(visual_img, f"Recognition result: {recognized_digit}", 
                        (visual_img.shape[1]//2 - 150, visual_img.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Display status information
        cv2.putText(visual_img, f"Reference number: {self.reference_number}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Match status
        if self.match_found:
            status_text = "Status: Match successful, approaching"
            status_color = (0, 255, 0)
        else:
            status_text = "Status: Recognizing..."
            status_color = (0, 165, 255)
        
        cv2.putText(visual_img, status_text, 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Simplified display, only showing recognition results
        if self.debug_images:
            cv2.imshow("Number Recognition", visual_img)
            cv2.waitKey(1)
        
        return recognized_digit, bounding_box
    
    def approach_target(self):
        """Approach the target - simplified version, only moving forward for a fixed time"""
        rospy.loginfo("🚗 Approaching target...")
        
        # Set approach start time
        self.approach_start_time = rospy.Time.now().to_sec()
        
        # Reset any previous movement commands
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd) 
        rospy.sleep(0.2)  # Brief pause to reset movement state
        
        # Initialize current velocity command
        self.current_cmd = Twist()
        self.current_cmd.linear.x = self.approach_speed
        self.current_cmd.angular.z = 0.0
        
        # Send the first movement command directly
        for _ in range(3):  # Send multiple times to ensure reception
            self.cmd_vel_pub.publish(self.current_cmd)
            rospy.sleep(0.05)
        
        rospy.loginfo(f"🚀 Starting to move forward at {self.approach_speed} m/s, will continue for {self.approach_duration} seconds")
    
    def run(self):
        """Run the main loop"""
        rospy.loginfo("Starting digit recognition task...")
        
        # Wait to receive reference number
        timeout = rospy.Time.now() + rospy.Duration(2)  # 2 seconds timeout
        while self.reference_number is None and not rospy.is_shutdown():
            if rospy.Time.now() > timeout:
                rospy.logwarn("Timeout waiting for reference number, using default value '4'")
                self.reference_number = "4"
                break
            rospy.loginfo("Waiting to receive reference number...")
            rospy.sleep(1)
        
        # Main loop
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            # If a matching number is found, wait for approach process to complete
            if self.match_found:
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - self.approach_start_time
                
                # Simplified version: only check if the predetermined time has been reached
                if elapsed_time > self.approach_duration:
                    # Time's up, stop the robot
                    self.current_cmd = Twist()  # Reset to zero velocity
                    for _ in range(3):  # Send multiple times to ensure stopping
                        self.cmd_vel_pub.publish(self.current_cmd)
                        rospy.sleep(0.1)
                    
                    rospy.loginfo("✅ Forward movement complete! Matching number found and approached")
                    rospy.signal_shutdown("Task completed")
                    break
                
                # Simple logging of current status
                if int(elapsed_time * 2) % 2 == 0:  # Log every 0.5 seconds
                    progress = int((elapsed_time / self.approach_duration) * 100)
                    rospy.loginfo(f"Forward progress: {progress}%")
            
            rate.sleep()
    
    def shutdown(self):
        """Clean up resources"""
        # Stop timer
        if hasattr(self, 'cmd_timer'):
            self.cmd_timer.shutdown()
        
        # Send stop command
        stop_cmd = Twist()
        for _ in range(3):  # Send stop command multiple times to ensure stopping
            self.cmd_vel_pub.publish(stop_cmd)
            rospy.sleep(0.1)
        
        cv2.destroyAllWindows()
        rospy.loginfo("Number detector closed")

if __name__ == '__main__':
    try:
        detector = SimplifiedNumberDetector()
        rospy.on_shutdown(detector.shutdown)
        detector.run()
    except rospy.ROSInterruptException:
        pass