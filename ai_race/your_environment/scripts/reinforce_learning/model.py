# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    CHOICES = [
            -1.0, -0.5, 0.0, 0.5, 1.0
            ]
    INPUT_HEIGHT = 240
    INPUT_WIDTH = 320
    INPUT_CHANNEL = 3

    def __init__(self):
        super(Model, self).__init__()
        num_outputs = len(self.CHOICES)
        self.conv1 = nn.Conv2d(self.INPUT_CHANNEL, 9, 3)
        self.bn1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear((self.INPUT_HEIGHT-4)*(self.INPUT_WIDTH-4)*32, 32)
        self.head = nn.Linear(32, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.head(x)
        return x

    def choice(self, action):
        return self.CHOICES[action]

    @property
    def input_size(self):
        return [1, self.INPUT_CHANNEL, self.INPUT_HEIGHT, self.INPUT_WIDTH]
