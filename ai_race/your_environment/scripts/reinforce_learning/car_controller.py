# -*- coding: utf-8 -*-
import Queue

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# from keyboard_con_pygame_videosave import INFERENCE_TIME


INFERENCE_TIME = 0.055

class CarController(object):
    SPEED_DIFF = 0.2
    ANGLE_DIFF = 1
    MAX_SPEED = 0.5
    MIN_SPEED = 0

    def __init__(self):
        rospy.init_node('car_controller', anonymous=True)
        self.twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self._callback)
        self.reset()
        self._image = None
        self.bridge = CvBridge()

    def reset(self):
        self.speed = 0
        self.angle = 0
        self.move(self.speed, self.angle)

    def preprocess(self, image):
        return self.bridge.imgmsg_to_cv2(image, "bgr8")

    def _callback(self, image):
        self._image = image

    @property
    def image(self):
        if self._image is None:
            return None
        image = self.preprocess(self._image)
        self._image = None
        return image

    def acceleration(self):
        self.speed += CarController.SPEED_DIFF
        if self.speed > CarController.MAX_SPEED:
            self.speed = CarController.MAX_SPEED

    def brake(self):
        self.speed -= CarController.SPEED_DIFF
        if self.speed < CarController.MIN_SPEED:
            self.speed = CarController.MIN_SPEED

    def steering(self, scale):
        self.angle = -1 * CarController.ANGLE_DIFF * scale

    def move(self, speed, angle):
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = angle
        rospy.sleep(INFERENCE_TIME)
        self.twist_pub.publish(twist)

    def step(self):
        raise NotImplementedError

    def _step(self):
        self.step()
        self.move(self.speed, self.angle)
        self.angle = 0

    def start(self, rate=10):
        r = rospy.Rate(rate)
        while not rospy.is_shutdown():
            print("step")
            self._step()
            r.sleep()
