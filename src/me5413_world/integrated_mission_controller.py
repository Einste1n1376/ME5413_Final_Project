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
        
        # Initialize publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.bridge_pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=1)
        self.recognized_number_pub = rospy.Publisher('/recognized_number', String, queue_size=1)
        
        # Initialize subscribers
        self.status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)
        self.number_sub = rospy.Subscriber('/recognized_number', String, self.number_callback)
        
        # Navigation waypoints list - before bridge crossing
        self.goals = [
            # First waypoint - avoid obstacles in the first area
            {'x': 16.0, 'y': -22.0, 'z': 0.0, 'yaw': math.pi},
            # Second waypoint - reach the "riverbank" to observe obstacles
            {'x': 12, 'y': -11.6, 'z': 0.0, 'yaw': math.pi},
        ]
        
        # After-bridge waypoints list
        self.after_bridge_goals = [
            # Four waypoints after crossing the bridge, for detecting numbers at different locations
            {'x': 3.5, 'y': -4.8, 'z': 0.0, 'yaw': math.pi},
            {'x': 3.5, 'y': -9.6, 'z': 0.0, 'yaw': 0.0},
            {'x': 3.5, 'y': -13.4, 'z': 0.0, 'yaw': math.pi},
            {'x': 3.5, 'y': -19.8, 'z': 0.0, 'yaw': 0.0}
        ]
        
        # Status variables
        self.current_goal_index = 0  
        self.navigation_completed = False
        self.number_recognized = False
        self.recognized_digits = []
        self.mission_stage = "NAVIGATION"  # Mission stages: NAVIGATION, NUMBER_IDENTIFICATION, BRIDGE_CROSSING, AFTER_BRIDGE
        self.bridge_process = None
        self.number_process = None
        self.auto_detector_process = None
        self.after_bridge_process = None
        self.reached_final_goal = False
        self.bridge_crossing_completed = False
        self.after_bridge_target_number = None  # Number to be identified after crossing the bridge
        self.after_bridge_completed = False  # Whether the after-bridge task is completed
        
        # Initialize subscriber
        self.amcl_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.current_robot_pose = None  # For storing the current robot position    

        # Wait for node initialization
        rospy.sleep(2)
        rospy.loginfo("Integrated mission controller initialized")

    def amcl_pose_callback(self, msg):
        """Store current robot position"""
        self.current_robot_pose = msg

    def publish_current_pose_as_initial(self):
        """Get current position and republish as initial pose to reduce position drift"""
        try:
            initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
            rospy.sleep(0.5)  # Ensure the publisher is initialized
            
            # Prioritize using actual position
            if self.current_robot_pose is not None:
                # Use the latest actual position
                initial_pose = PoseWithCovarianceStamped()
                initial_pose.header = self.current_robot_pose.header
                initial_pose.pose = self.current_robot_pose.pose
                
                initial_pose_pub.publish(initial_pose)
                rospy.loginfo("Republishing current actual position as initial pose")
                rospy.sleep(1)  # Wait for pose initialization
                return
            
            # Set current position as initial pose
            initial_pose = PoseWithCovarianceStamped()
            initial_pose.header.frame_id = "map"
            initial_pose.header.stamp = rospy.Time.now()
            
            # Choose target point based on current mission stage
            goal_list = self.goals if self.mission_stage != "AFTER_BRIDGE" else self.after_bridge_goals
            
            # Fix: Use previous waypoint position (current reached position) as current position
            previous_goal_index = self.current_goal_index - 1
            if previous_goal_index >= 0:
                # Use previous waypoint position
                previous_goal = goal_list[previous_goal_index]
                initial_pose.pose.pose.position.x = previous_goal['x'] 
                initial_pose.pose.pose.position.y = previous_goal['y']
                initial_pose.pose.pose.position.z = 0.0
                yaw = previous_goal['yaw']
            else:
                # If it's the first waypoint, use starting position (or first waypoint)
                current_goal = goal_list[self.current_goal_index]
                initial_pose.pose.pose.position.x = current_goal['x'] 
                initial_pose.pose.pose.position.y = current_goal['y']
                initial_pose.pose.pose.position.z = 0.0
                yaw = current_goal['yaw']
            
            initial_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
            initial_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
            
            # Set covariance matrix
            initial_pose.pose.covariance = [0.0] * 36
            initial_pose.pose.covariance[0] = 0.25  # x-direction covariance
            initial_pose.pose.covariance[7] = 0.25  # y-direction covariance
            initial_pose.pose.covariance[35] = 0.06853891945200942  # yaw covariance
            
            initial_pose_pub.publish(initial_pose)
            rospy.loginfo("Republishing current position as initial pose")
            rospy.sleep(1)  # Wait for pose initialization
        except Exception as e:
            rospy.logwarn(f"Failed to republish initial pose: {e}")

    def publish_next_goal(self):
        """Publish next navigation waypoint"""
        # Choose waypoint list based on current mission stage
        goal_list = self.goals if self.mission_stage != "AFTER_BRIDGE" else self.after_bridge_goals
        
        if self.current_goal_index >= len(goal_list):
            if self.mission_stage == "NAVIGATION":
                rospy.loginfo("All waypoints in navigation stage completed")
                self.navigation_completed = True
            elif self.mission_stage == "AFTER_BRIDGE":
                rospy.loginfo("All waypoints in after-bridge stage completed")
                self.after_bridge_completed = True
            return
        
        goal_data = goal_list[self.current_goal_index]
        
        # Create waypoint message
        goal = PoseStamped()
        goal.header.frame_id = "map"  
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = goal_data['x']
        goal.pose.position.y = goal_data['y']
        goal.pose.position.z = goal_data['z']

        yaw = goal_data['yaw']
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # Publish waypoint
        stage_name = "Navigation" if self.mission_stage == "NAVIGATION" else "After-bridge"
        rospy.loginfo(f"Publishing {stage_name} waypoint {self.current_goal_index + 1}: x={goal_data['x']}, y={goal_data['y']}")
        self.goal_pub.publish(goal)
        
    def status_callback(self, msg):
        """Process navigation status feedback"""
        if not msg.status_list or (self.mission_stage != "NAVIGATION" and self.mission_stage != "AFTER_BRIDGE"):
            return
            
        # Get latest status
        status = msg.status_list[-1].status
        
        # Status 3 means goal reached
        if status == 3:  # SUCCEEDED
            rospy.loginfo(f"Reached waypoint {self.current_goal_index + 1}")
            
            if self.mission_stage == "NAVIGATION":
                # If this is the last waypoint, wait for stability before starting number identification
                if self.current_goal_index == len(self.goals) - 1:
                    # Set flag indicating final waypoint reached
                    self.reached_final_goal = True
                    # Critical fix: Add delay to ensure robot is fully stable before starting number identification
                    rospy.loginfo("Reached riverbank, waiting for position to stabilize...")
                    rospy.sleep(20)  # Wait 20 seconds for robot to stabilize
                    rospy.loginfo("Now starting number identification...")
                    self.start_number_identification()
                else:
                    # Proceed to next waypoint
                    self.current_goal_index += 1
                    # First republish current pose to reduce drift
                    self.publish_current_pose_as_initial()
                    rospy.sleep(5)  # Wait for pose to stabilize
                    self.publish_next_goal()
            
            elif self.mission_stage == "AFTER_BRIDGE":
                # After-bridge waypoint navigation
                rospy.loginfo(f"After bridge, reached waypoint {self.current_goal_index + 1}")
                
                # Start number identification node to scan for numbers at this location
                rospy.loginfo("Starting to scan for numbers at current location...")
                self.start_after_bridge_recognition()
                
                # Wait a few seconds to ensure number identification node has time to run
                rospy.sleep(10)
                
                # If this is the last waypoint, mark after-bridge task as completed
                if self.current_goal_index == len(self.after_bridge_goals) - 1:
                    rospy.loginfo("Reached last after-bridge waypoint")
                    self.after_bridge_completed = True
                else:
                    # Stop current number identification node
                    self.stop_after_bridge_recognition()
                    
                    # Proceed to next waypoint
                    self.current_goal_index += 1
                    # First republish current pose to reduce drift
                    self.publish_current_pose_as_initial()
                    rospy.sleep(2)  # Wait for pose to stabilize
                    self.publish_next_goal()
    
    def number_callback(self, msg):
        """Process identified number"""
        digit = msg.data
        rospy.loginfo(f"Received identified number: {digit}")
        
        if self.mission_stage == "NUMBER_IDENTIFICATION":
            # Add identified number to the list
            self.recognized_digits.append(digit)
            
            # If enough number samples collected, determine the most frequent number
            if len(self.recognized_digits) >= 10:
                self.number_recognized = True
                
                # Count occurrences of each number
                digit_counts = {}
                for d in self.recognized_digits:
                    if d in digit_counts:
                        digit_counts[d] += 1
                    else:
                        digit_counts[d] = 1
                
                # Find the most frequent number (modified to most frequent instead of least)
                most_common_digit = max(digit_counts.items(), key=lambda x: x[1])
                
                # Save this number for use in after-bridge task
                self.after_bridge_target_number = most_common_digit[0]
                
                rospy.loginfo(f"Determined most common number is: {self.after_bridge_target_number} (occurred {most_common_digit[1]} times)")
                rospy.loginfo("Number identification stage completed, preparing to cross bridge...")
                
                # Stop number identification node
                if self.number_process:
                    try:
                        self.number_process.terminate()
                        rospy.loginfo("Number identification node stopped")
                    except:
                        rospy.logwarn("Failed to stop number identification node")
                
                # Start bridge crossing task
                self.start_bridge_crossing()
    
    def start_number_identification(self):
        """Start number identification node"""
        self.mission_stage = "NUMBER_IDENTIFICATION"
        rospy.loginfo("Starting number identification node...")
        
        try:
            # Start number identification node
            self.number_process = subprocess.Popen(
                ["rosrun", "me5413_world", "number_identification.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rospy.loginfo("Number identification node started")
        except Exception as e:
            rospy.logerr(f"Failed to start number identification node: {e}")
            # If startup fails, proceed directly to bridge crossing stage
            self.start_bridge_crossing()
    
    def start_bridge_crossing(self):
        """Start bridge crossing task"""
        self.mission_stage = "BRIDGE_CROSSING"
        rospy.loginfo("Starting orange cone detection node...")
        
        try:
            # Start orange cone detection node
            self.auto_detector_process = subprocess.Popen(
                ["rosrun", "me5413_world", "auto_detector_controller.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rospy.loginfo("Orange cone detection node started")
            
            # Listen for bridge crossing completion signal
            # In actual application, a subscriber should be added here to receive bridge crossing completion message
            # Simplified handling: assume bridge crossing completes after 60 seconds
            rospy.Timer(rospy.Duration(60), self.bridge_crossing_completed_callback, oneshot=True)
            
        except Exception as e:
            rospy.logerr(f"Failed to start orange cone detection node: {e}")
            rospy.signal_shutdown("Task execution failed")
    
    def bridge_crossing_completed_callback(self, event):
        """Callback after bridge crossing is completed"""
        rospy.loginfo("Bridge crossing task completed, preparing to enter after-bridge stage...")
        
        # Stop bridge crossing related nodes
        if self.auto_detector_process:
            try:
                self.auto_detector_process.terminate()
                rospy.loginfo("Orange cone detection node stopped")
            except:
                rospy.logwarn("Failed to stop orange cone detection node")
        
        # Mark bridge crossing as completed
        self.bridge_crossing_completed = True
        
        # Set after-bridge stage
        self.start_after_bridge_phase()
    
    def start_after_bridge_phase(self):
        """Start after-bridge task stage"""
        self.mission_stage = "AFTER_BRIDGE"
        self.current_goal_index = 0  # Reset waypoint index
        
        rospy.loginfo(f"Starting after-bridge task stage, target is to find number: {self.after_bridge_target_number}")
        
        # Ensure there is a target number
        if not self.after_bridge_target_number:
            self.after_bridge_target_number = "5"  # Default value
            rospy.logwarn(f"No target number identified, using default value: {self.after_bridge_target_number}")
        
        # Start navigation to first after-bridge waypoint
        self.publish_next_goal()
    
    def start_after_bridge_recognition(self):
        """Start after-bridge number identification node"""
        rospy.loginfo(f"Starting after-bridge number identification node, looking for number: {self.after_bridge_target_number}")
        
        # First publish target number
        self.recognized_number_pub.publish(self.after_bridge_target_number)
        rospy.sleep(1)  # Ensure message is published
        
        try:
            # Start number identification node
            self.after_bridge_process = subprocess.Popen(
                ["rosrun", "me5413_world", "after_bridge_front_recognize.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rospy.loginfo("After-bridge number identification node started")
            
            # Set timeout mechanism
            rospy.Timer(rospy.Duration(30), self.after_bridge_recognition_timeout, oneshot=True)
            
        except Exception as e:
            rospy.logerr(f"Failed to start after-bridge number identification node: {e}")
            # If failed, continue to next waypoint
            if self.current_goal_index < len(self.after_bridge_goals) - 1:
                self.current_goal_index += 1
                self.publish_next_goal()
            else:
                self.after_bridge_completed = True
    
    def stop_after_bridge_recognition(self):
        """Stop after-bridge number identification node"""
        if self.after_bridge_process:
            try:
                self.after_bridge_process.terminate()
                rospy.loginfo("After-bridge number identification node stopped")
                self.after_bridge_process = None
            except:
                rospy.logwarn("Failed to stop after-bridge number identification node")
    
    def after_bridge_recognition_timeout(self, event):
        """Handle number identification timeout"""
        # Check if node is still running
        if self.after_bridge_process and self.after_bridge_process.poll() is None:
            rospy.loginfo("Target number not found at current location, continuing to next location")
            self.stop_after_bridge_recognition()
            
            # If not the last waypoint, continue to next
            if self.current_goal_index < len(self.after_bridge_goals) - 1:
                self.current_goal_index += 1
                self.publish_next_goal()
            else:
                rospy.loginfo("Checked all locations, target number not found")
                self.after_bridge_completed = True
    
    def start_mission(self):
        """Start the entire mission"""
        rospy.loginfo("Starting navigation task...")
        self.publish_next_goal()
        
        # Enter main loop
        rate = rospy.Rate(1)  # 1Hz
        wait_counter = 0
        stabilization_time = 10  # 10 seconds stabilization time
        
        while not rospy.is_shutdown():
            # If reached final waypoint but navigation not yet marked as completed, wait for stabilization
            if self.reached_final_goal and not self.navigation_completed:
                wait_counter += 1
                rospy.loginfo(f"Waiting for position to stabilize ({wait_counter}/{stabilization_time} seconds)...")
                
                if wait_counter >= stabilization_time:
                    rospy.loginfo("Position sufficiently stabilized, navigation stage completed")
                    self.navigation_completed = True
                    wait_counter = 0
            
            # If navigation completed and still in navigation stage, start number identification
            elif self.mission_stage == "NAVIGATION" and self.navigation_completed:
                rospy.loginfo("Formally starting number identification task...")
                self.start_number_identification()
            
            # If number identification completed, start bridge crossing
            elif self.mission_stage == "NUMBER_IDENTIFICATION" and self.number_recognized:
                rospy.loginfo("Number identification task completed, preparing to cross bridge...")
                self.start_bridge_crossing()
            
            # If bridge crossing completed, start after-bridge task
            elif self.mission_stage == "BRIDGE_CROSSING" and self.bridge_crossing_completed:
                rospy.loginfo("Bridge crossing task completed, starting after-bridge task...")
                self.start_after_bridge_phase()
            
            # Check if after-bridge task is completed
            elif self.mission_stage == "AFTER_BRIDGE":
                # Check if after_bridge_process has terminated (may have found target number)
                if self.after_bridge_process and self.after_bridge_process.poll() is not None:
                    rospy.loginfo("After-bridge number identification node has exited, may have found target number")
                    self.after_bridge_process = None
                    self.after_bridge_completed = True
                
                # If after-bridge task completed, terminate program
                if self.after_bridge_completed:
                    rospy.loginfo("✅ All tasks completed, program will exit")
                    rospy.signal_shutdown("Task completed")
                    break
            
            rate.sleep()
    
    def shutdown(self):
        """Clean up processes and shut down properly"""
        rospy.loginfo("Shutting down all processes...")
        
        # Close all child processes
        for process in [self.number_process, self.auto_detector_process, self.after_bridge_process]:
            if process:
                try:
                    process.terminate()
                except:
                    pass
        
        rospy.loginfo("Integrated mission controller shut down")

if __name__ == '__main__':
    try:
        controller = IntegratedMissionController()
        rospy.on_shutdown(controller.shutdown)
        controller.start_mission()
    except rospy.ROSInterruptException:
        pass