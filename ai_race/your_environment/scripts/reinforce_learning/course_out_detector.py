# -*- coding: utf-8 -*-
import math

import rospy
from gazebo_msgs.msg import ModelStates


class CourseOutDetector(object):
    # INNER_X = 1.1
    # INNER_Y = 1
    # OUTER_X = 3.0
    # OUTER_Y = 2.5
    # RADIUS = 0.5
    INNER_X = 1.3
    INNER_Y = 1.9
    OUTER_X = 2.1
    OUTER_Y = 2.7

    def __init__(self):
        # rospy.init_node("course_out_detector", anonymous=True)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback, queue_size=1)
        self.data = None

    def callback(self, data):
        self.data = data

    def get_position(self, data):
        pos = data.name.index('wheel_robot')
        x = data.pose[pos].position.x
        y = data.pose[pos].position.y
        return x, y

    @property
    def course_outed(self):
        data = self.data
        return self.is_course_outed(data)

    def is_course_outed(self, data):
        if data is None:
            return False
        x, y = self.get_position(data)
        if abs(x) < self.INNER_X and abs(y) < self.INNER_Y:
            return True
        if abs(x) > self.OUTER_X or abs(y) > self.OUTER_Y:
            return True
        return False
        # if abs(x) >= self.OUTER_X or abs(y) >= self.OUTER_Y:
        #     return True
        # elif abs(y) <= self.INNER_Y:
        #     if abs(x) <= self.INNER_X:
        #         return True
        # elif y > self.INNER_Y:
        #     if abs(x) < (self.INNER_X - self.RADIUS) \
        #             and (y < (self.INNER_Y + self.RADIUS)):
        #         return True
        #     elif x >= (self.INNER_X - self.RADIUS)\
        #             and (x - (self.INNER_X - self.RADIUS)) ** 2 + (y - self.INNER_Y) ** 2 < self.RADIUS ** 2:
        #         return True
        #     elif x <= -(self.INNER_X - self.RADIUS)\
        #             and (x + (self.INNER_X - self.RADIUS)) ** 2 + (y - self.INNER_Y) ** 2 < self.RADIUS ** 2:
        #         return True
        # elif y < -self.INNER_Y:
        #     if abs(x) < (self.INNER_X - self.RADIUS) \
        #             and (y > -(self.INNER_Y + self.RADIUS)):
        #         return True
        #     elif x >= (self.INNER_X - self.RADIUS)\
        #             and (x - (self.INNER_X - self.RADIUS)) ** 2 + (y + self.INNER_Y) ** 2 < self.RADIUS ** 2:
        #         return True
        #     elif x <= -(self.INNER_X - self.RADIUS)\
        #             and (x + (self.INNER_X - self.RADIUS)) ** 2 + (y + self.INNER_Y) ** 2 < self.RADIUS ** 2:
        #         return True
        # return False

    @property
    def distance(self):
        data = self.data
        course_outed = self.is_course_outed(data)
        if course_outed:
            return -1
        else:
            x, y = self.get_position(data)
            if abs(y) < self.INNER_Y:
                return abs(x) - self.INNER_X
            elif abs(x) < self.INNER_X:
                return abs(y) - self.INNER_Y
            else:
                return math.sqrt((abs(x) - self.INNER_X)**2 + (abs(y) - self.INNER_Y)**2)
