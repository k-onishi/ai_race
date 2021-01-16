# -*- coding: utf-8 -*-
import json
import os
import requests
import sys

import torch
import rospy
from sensor_msgs.msg import Image

from agent import DeepQNetworkAgent
from car_controller import CarController
from course_out_detector import CourseOutDetector
from model import CustomModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from reset_sample import (
        Init, Start, ManualRecovery,
        JUDGESERVER_GETSTATE_URL
)


def get_dict_value(dict_data, keys, default_value=None):
    data = dict_data
    for key in keys:
        if key in data:
            data = data[key]
        else:
            return default_value
    return data

class GameState(object):
    def __init__(self, json_data=None):
        if json_data is None:
            dict_data = {}
        else:
            dict_data = json.loads(json_data)
        self._is_courseout = int(get_dict_value(dict_data, ["judge_info", "is_courseout"], 334))

    @property
    def is_courseout(self):
        return self._is_courseout

class ModelLearner(CarController):
    def __init__(self, update_teacher_interval=10, num_epoch=20,
            model_path=None, model_kwargs={}):
        self.model_kwargs = model_kwargs
        self.agent = DeepQNetworkAgent(
            CustomModel,
            height=120,
            width=320,
            max_experience=500,
            batch_size=16,
            input_size=[1, 3, 240, 320],
            model_path=model_path,
            model_kwargs=model_kwargs)
        print("model: {}".format(self.agent.model))
        self.count = 0
        self.num_epoch = num_epoch
        self.episode_count = 0
        print("Initialize detector")
        self.course_out_detector = CourseOutDetector()
        self.previous_image = None
        self.previous_action = None
        self.update_teacher_interval = update_teacher_interval
        self.done = False
        super(ModelLearner, self).__init__()

    def preprocess(self, image):
        image = super(ModelLearner, self).preprocess(image)
        return self.agent.preprocess(image)[:, :, 120:, :]

    def _callback(self, image):
        if self.done:
            return

        super(ModelLearner, self)._callback(image)
        current_image = self.image
        self.count += 1
        self.acceleration()
        action = self.agent.policy(current_image)
        self.steering(self.agent.choice(action))
        self.move(self.speed, self.angle)

        distance = self.course_out_detector.distance
        if distance < 0:
            current_image = None
            reward = -1.0
            self.done = True
        else:
            # distance
            # max: sqrt(0.6^2 + 0.6^2) = 0.8485... = 0.90
            reward = (0.9 - distance) // 0.3 / 2.0
            self.done = False
        if self.previous_image is not None:
            self.agent.add_experience(
                self.previous_image,
                self.previous_action,
                current_image,
                torch.tensor([reward])
            )
        self.previous_image = current_image
        self.previous_action = action

    @property
    def status(self):
        res = requests.get(
            JUDGESERVER_GETSTATE_URL,
        ).json()["judge_info"]
        return res

    def step(self):
        status = self.status
        done = self.course_out_detector.course_outed
        if done:
            self.image_sub.unregister()
            self.episode_count += 1
            print("episode: {}\tLAP: {}\tstep count: {}".format(
                self.episode_count, status["lap_count"], self.count))
            self.reset()
            for epoch in range(self.num_epoch):
                loss = self.agent.update()
                if loss is None:
                    break
            print("Loss: {}".format(loss))
            if self.episode_count % self.update_teacher_interval == 0:
                self.agent.update_teacher()
            Start()
            self.image_sub = rospy.Subscriber('front_camera/image_raw', Image, self._callback)

    def start(self):
        Start()
        super(ModelLearner, self).start()

    def reset(self):
        super(ModelLearner, self).reset()
        self.count = 0
        self.previous_image = None
        Init()
        ManualRecovery()
        self.done = False
        rospy.sleep(1.0)


if __name__ == '__main__':
    try:
        Init()
        print("Initialize")
        learner = ModelLearner(
            model_path="/home/jetson/save_dir/model.pth",
            update_teacher_interval=2,
            model_kwargs={
                "conv_channels": [16, 32, 32],
                "kernel_size": 5,
                "stride": 2,
            })
        learner.start()
    except rospy.ROSInterruptException:
        print("ModelLearner stopped")
        learner.reset()
