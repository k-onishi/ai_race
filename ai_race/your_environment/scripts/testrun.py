#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from sensor_msgs.msg import Image

import time
import sys
import os

class keyboardController:
    def __init__(self):
        #ノード初期設定、10fpsのタイマー付きパブリッシャー
        rospy.init_node('keyboard_con_node', anonymous=True)
        rospy.sleep(1.0)
        self.prev_x = 0
        self.prev_z = 0
        self.start_time = rospy.Time.now().to_sec()
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self.callback)
        self.coff_angularz = 1
        #self.start_time = time.time()


    def call_cmd_publish(self, x_velocity, z_angular):
        twist = Twist()
        twist.linear.x = x_velocity  # m/s
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = z_angular  # rad
        self.twist_pub.publish(twist)
    
    def callback(self, event):

        x = 1.6   # x velocity (m/s)
        z = 0     # z angular (rad)

        curve_sec = 3.14/2 - 0.24 #1.33
        
        t1 = 0.80              # go straight
        t2 = t1 + curve_sec    # curve
        t3 = t2 + 0.5          # go straight
        t4 = t3 + curve_sec    # curve
        t5 = t4 + 1.25         # go straight
        t6 = t5 + curve_sec + 0.05    # curve
        t7 = t6 + 0.55           # go straight
        t8 = t7 + curve_sec - 0.00    # curve
        t9 = t8 + 0.30         # go straight
        loop_time = t9

        #elapsed_time = time.time() - self.start_time
        elapsed_time = rospy.Time.now().to_sec() - self.start_time
        elapsed_time = elapsed_time % t9

        if elapsed_time < t1:
            z = 0
        elif elapsed_time < t2:
            z = 1 * self.coff_angularz # left
        elif elapsed_time < t3:
            z = 0
        elif elapsed_time < t4:
            z = 1 * self.coff_angularz # left
        elif elapsed_time < t5:
            z = 0
        elif elapsed_time < t6:
            z = 1 * self.coff_angularz # left
        elif elapsed_time < t7:
            z = 0
        elif elapsed_time < t8:
            z = 1 * self.coff_angularz # left
        elif elapsed_time < t9:
            z = 0
        else:
            z = 0

        if self.prev_x != x or self.prev_z != z:
            print("---")
            print(elapsed_time)
            print(self.start_time)
            print("publish x:"+ str(x))
            print("publish z:"+ str(z))
        self.prev_x = x
        self.prev_z = z

        self.call_cmd_publish(x, z)
        
if __name__ == '__main__':
    kc = keyboardController()
    rospy.spin()
