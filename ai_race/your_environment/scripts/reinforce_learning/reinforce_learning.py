import json
import os
import requests
import sys
import time

# import numpy as np
import torch
import rospy
# from std_msgs.msg import String

from agent import DeepQNetworkAgent
from car_controller import CarController
from course_out_detector import CourseOutDetector
from model import Model

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from reset_sample import (
        Init, Start, Stop, ManualRecovery,
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
    def __init__(self, update_teacher_count=10):
        super(ModelLearner, self).__init__()
        print("Initialize agent")
        self.agent = DeepQNetworkAgent(
                Model,
                max_experience=10,
                batch_size=4
        )
        self.count = 0
        self.episode_count = 0
        print("Initialize detector")
        self.course_out_detector = CourseOutDetector()
        self.previous_image = None
        self.previous_action = None
        self.update_teacher_count = update_teacher_count

    def preprocess(self, image):
        image = super(ModelLearner, self).preprocess(image)
        return self.agent.preprocess(image)

    @property
    def state(self):
        res = requests.get(
            JUDGESERVER_GETSTATE_URL,
        ).json()["judge_info"]
        return res

    def step(self):
        current_image = self.image
        if current_image is None:
            return
        self.count += 1
        self.acceleration()
        state = self.state
        print("lap: {}\tcourseout: {}".format(
            state["lap_count"], state["is_courseout"] == 1))
        action = self.agent.policy(current_image)
        self.steering(self.agent.choice(action))
        done = self.course_out_detector.course_outed
        if done:
            current_image = None
            reward = -1
        else:
            reward = self.count // 10 + 1
        if self.previous_image is not None:
            self.agent.add_experience(
                    self.previous_image,
                    self.previous_action,
                    current_image,
                    torch.tensor([reward], device=self.agent.device)
            )
        self.previous_image = current_image
        self.previous_action = action
        self.agent.update()
        if done:
            print("Course out")
            self.episode_count += 1
            if self.episode_count % self.update_teacher_count == 0:
                self.agent.update_teacher()
            self.reset()
            ManualRecovery()
            time.sleep(3)
            Start()

    def start(self):
        super(ModelLearner, self).start()
        Start()

    def reset(self):
        super(ModelLearner, self).reset()
        Stop()
        self.count = 0


if __name__ == '__main__':
    try:
        Init()
        print("Initialize")
        learner = ModelLearner()
        print("Start")
        learner.start()
    except rospy.ROSInterruptException:
        print("ModelLearner stopped")
        learner.reset()
