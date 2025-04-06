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
        
        # Reduce callback frequency to reduce computational burden
        self.image_sub = rospy.Subscriber("/second/img_second", Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.number_pub = rospy.Publisher("/recognized_number", String, queue_size=1)
        
        # Process control
        self.processing = False
        self.skip_count = 0
        self.process_every_n_frames = 10  # Process once every 30 frames
        
        # History record for improved stability
        self.last_digits = []
        self.max_history = 5
        
        rospy.loginfo("Number counting node initialized")
        
    def image_callback(self, msg):
        # Skip some frames to reduce CPU load
        self.skip_count += 1
        if self.skip_count % self.process_every_n_frames != 0:
            return
        
        if self.processing:
            return  # Skip this frame if the previous frame is still being processed
        
        self.processing = True
        try:
            # Convert ROS image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Rotate 90 degrees clockwise
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Crop the image, keeping only the middle 3/5 portion
            width = cv_image.shape[1]
            left_cut = width // 5  # Left 1/5 will be cropped
            right_cut = width - (width // 5)  # Right 1/5 will be cropped
            cv_image = cv_image[:, left_cut:right_cut]  # Keep the middle 3/5
            
            # Process the image
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
        finally:
            self.processing = False
    
    def non_max_suppression(self, boxes, overlapThresh=0.3):
        """
        Apply non-maximum suppression (NMS) algorithm to filter overlapping bounding boxes
        
        Parameters:
            boxes: List of candidate boxes in format (x, y, w, h, digit, confidence)
            overlapThresh: Overlap threshold, boxes exceeding this threshold will be filtered out
            
        Returns:
            Filtered list of bounding boxes
        """
        # If no boxes, return empty list
        if len(boxes) == 0:
            return []
        
        # Initialize list of selected indices
        pick = []
        
        # Extract coordinates
        x = np.array([box[0] for box in boxes])
        y = np.array([box[1] for box in boxes])
        w = np.array([box[2] for box in boxes])
        h = np.array([box[3] for box in boxes])
        
        # Calculate area and bottom-right coordinates for each box
        area = w * h
        xx2 = x + w
        yy2 = y + h
        
        # Sort by area (small to large)
        idxs = np.argsort(area)
        
        # Process until no bounding boxes remain
        while len(idxs) > 0:
            # Take the last one (largest area)
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find all boxes that overlap with current box
            xx1 = np.maximum(x[i], x[idxs[:last]])
            yy1 = np.maximum(y[i], y[idxs[:last]])
            xx2_overlap = np.minimum(xx2[i], xx2[idxs[:last]])
            yy2_overlap = np.minimum(yy2[i], yy2[idxs[:last]])
            
            # Calculate width and height of overlap area
            w_overlap = np.maximum(0, xx2_overlap - xx1)
            h_overlap = np.maximum(0, yy2_overlap - yy1)
            
            # Calculate ratio of overlap area to smaller box area
            overlap = (w_overlap * h_overlap) / area[idxs[:last]]
            
            # Delete overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
        # Return kept boxes
        return [boxes[i] for i in pick]
    
    def process_image(self, image_color):
        # Display original image
        original_img = image_color.copy()
        
        # Convert to grayscale
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_gray = clahe.apply(image_gray)
        
        # Gaussian blur for noise reduction
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        
        # Adaptive threshold processing
        thresh = cv2.adaptiveThreshold(
            image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Use edge detection to enhance digit contours
        edges = cv2.Canny(image_gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Candidate box list, format: (x, y, w, h, digit, confidence)
        digit_candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            area = w * h
            
            # Filter out areas that are too small or too large
            min_area = 100  # Minimum area
            max_area = image_color.shape[0] * image_color.shape[1] // 10  # Maximum area is 1/10 of the image
            
            if min_area < area < max_area and 0.2 < aspect_ratio < 2.0:
                # Extract region of interest
                roi = image_gray[y:y+h, x:x+w]
                
                # Check pixel density in the region
                pixel_density = np.sum(thresh[y:y+h, x:x+w] > 0) / (w * h)
                
                # If density is too low or too high, it might not be a digit
                if 0.1 < pixel_density < 0.9:
                    # Resize ROI to appropriate range
                    target_height = 100
                    scale = target_height / h
                    target_width = int(w * scale)
                    roi_resized = cv2.resize(roi, (target_width, target_height))
                    
                    # Enhance contrast
                    roi_enhanced = cv2.convertScaleAbs(roi_resized, alpha=1.5, beta=10)
                    
                    # Use tesseract to recognize digits - only recognize 1-9
                    text = pytesseract.image_to_string(roi_enhanced, config='--oem 3 --psm 10 -c tessedit_char_whitelist=123456789')
                    text = ''.join(filter(lambda c: c in '123456789', text))
                    
                    if text and len(text) == 1:
                        # Add confidence estimate - using pixel density as a simple confidence indicator
                        confidence = pixel_density
                        digit_candidates.append((x, y, w, h, text, confidence))
        
        # Apply non-maximum suppression to filter overlapping boxes
        filtered_candidates = self.non_max_suppression(digit_candidates, overlapThresh=0.3)
        
        # Count occurrences of filtered digits
        digit_counts = defaultdict(int)
        digit_boxes = defaultdict(list)
        
        # Display filtered candidate boxes on the image
        for x, y, w, h, digit, confidence in filtered_candidates:
            digit_counts[digit] += 1
            digit_boxes[digit].append((x, y, w, h))
            
            # Mark recognized digits on the image
            cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(original_img, f"{digit}", 
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Select the digit with the lowest occurrence
        recognized_digit = None
        if digit_counts:
            # Find digit with lowest occurrence count
            min_digit = min(digit_counts.items(), key=lambda x: x[1])
            digit = min_digit[0]
            count = min_digit[1]
            
            recognized_digit = digit
            
            # Mark the digit with the lowest occurrence count on the image
            if digit in digit_boxes and digit_boxes[digit]:
                # Get the first bounding box
                x, y, w, h = digit_boxes[digit][0]
                
                # Mark the least common digit
                cv2.rectangle(original_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(original_img, f"Least Common: {digit} (count: {count})", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Add to history record
            self.last_digits.append(recognized_digit)
            if len(self.last_digits) > self.max_history:
                self.last_digits.pop(0)
            
            # Use the most frequent digit in history as final recognition result for improved stability
            history_counts = defaultdict(int)
            for d in self.last_digits:
                history_counts[d] += 1
            
            final_digit = max(history_counts.items(), key=lambda x: x[1])[0]
            
            rospy.loginfo(f"Digit with lowest occurrence in current frame: {recognized_digit} (count: {count})")
            rospy.loginfo(f"Stabilized digit from history: {final_digit} (history: {self.last_digits})")
            
            # Publish recognition result
            self.number_pub.publish(recognized_digit)
        
        # Display digit count results
        count_text = ", ".join([f"{d}: {c}" for d, c in sorted(digit_counts.items())])
        cv2.putText(original_img, f"Counts: {count_text}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display recognition results
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