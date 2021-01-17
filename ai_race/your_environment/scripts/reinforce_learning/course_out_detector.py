# -*- coding: utf-8 -*-
import math

import rospy
from gazebo_msgs.msg import ModelStates


class CourseOutDetector(object):
    WIDTH = 0.4
    HEIGHT = 1.0
    RADIUS = 1.0
    PATH_WIDTH = 0.8

    def __init__(self):
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
        distance = self.distance
        if distance >= 0 and distance <= self.PATH_WIDTH:
            return False
        else:
            return True

    @property
    def distance(self):
        data = self.data
        _x, _y = self.get_position(data)
        # 第一象限で考える
        x, y = abs(_x), abs(_y)
        if (x >= self.WIDTH + self.RADIUS) and (y <= self.HEIGHT):
            # コースの右側にある場合
            return x - (self.WIDTH + self.RADIUS)
        elif (x <= self.WIDTH) and (y >= self.HEIGHT + self.WIDTH):
            # コースの上側にある場合
            return y - (self.HEIGHT + self.RADIUS)
        elif (x >= self.WIDTH) and (y >= self.HEIGHT):
            # コースのコーナーにいる場合
            return math.sqrt((x - self.WIDTH)**2 + (y - self.HEIGHT) ** 2) - self.RADIUS
        else:
            # コースの中に入っている場合。めんどくさいので-1を返します
            return -1
