#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatusArray

class NavigationController:
    def __init__(self):
        rospy.init_node('goal_nav_node', anonymous=True)
        
        # Initialize publisher
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # Initialize subscriber - monitor navigation status
        self.status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)
        
        # Navigation waypoints list
        self.goals = [
            # First waypoint
            {'x': 16.0, 'y': -22.0, 'z': 0.0, 'yaw': math.pi},
            # Second waypoint
            {'x': 8.9, 'y': -11.6, 'z': 0.0, 'yaw': 0.0}
        ]
        
        # Current waypoint index
        self.current_goal_index = 0
        
        # Flag to mark if navigation is completed
        self.navigation_completed = False
        
        # Wait for node initialization
        rospy.sleep(2)
        rospy.loginfo("Navigation controller initialized")

    def publish_initial_pose(self):
        """Publish initial pose"""
        initial_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        rospy.sleep(1)  # Ensure the publisher is fully initialized

        # Set initial pose
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = "map"
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.pose.pose.position.x = 0.0  # Initial position x
        initial_pose.pose.pose.position.y = 0.0  # Initial position y
        initial_pose.pose.pose.position.z = 0.0  # Initial position z

        yaw = 0.0  # Initial orientation (radians)
        initial_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        initial_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Set covariance matrix
        initial_pose.pose.covariance = [0.0] * 36
        initial_pose.pose.covariance[0] = 0.25  # x direction covariance
        initial_pose.pose.covariance[7] = 0.25  # y direction covariance
        initial_pose.pose.covariance[35] = 0.06853891945200942  # yaw covariance

        initial_pose_pub.publish(initial_pose)
        rospy.loginfo("Published initial pose")

    def publish_next_goal(self):
        """Publish next navigation waypoint"""
        if self.current_goal_index >= len(self.goals):
            rospy.loginfo("All navigation goals completed")
            self.navigation_completed = True
            return
        
        goal_data = self.goals[self.current_goal_index]
        
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
        rospy.loginfo(f"Publishing goal {self.current_goal_index + 1}: x={goal_data['x']}, y={goal_data['y']}")
        self.goal_pub.publish(goal)
        
    def status_callback(self, msg):
        """Handle navigation status feedback"""
        if not msg.status_list:
            return
            
        # Get latest status
        status = msg.status_list[-1].status
        
        # Status 3 means goal reached
        if status == 3:  # SUCCEEDED
            rospy.loginfo(f"Reached goal {self.current_goal_index + 1}")
            
            # If this is the last waypoint
            if self.current_goal_index == len(self.goals) - 1:
                rospy.loginfo("Reached final destination, navigation completed")
                self.navigation_completed = True
            else:
                # Move to the next waypoint
                self.current_goal_index += 1
                rospy.sleep(2)  # Brief wait
                self.publish_next_goal()
                
    def start_navigation(self):
        """Start navigation task"""
        # Optional: publish initial pose
        # self.publish_initial_pose()
        
        # Publish first waypoint
        self.publish_next_goal()
        
        # Enter main loop
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