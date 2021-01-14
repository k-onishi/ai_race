# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    MID_CHANNEL = 16

    def __init__(self, choices,
            input_height=240, input_width=320, input_channels=3):
        super(Model, self).__init__()
        self.CHOICES = choices
        self.INPUT_HEIGHT = input_height
        self.INPUT_WIDTH = input_width
        self.INPUT_CHANNEL = input_channels

        num_outputs = len(self.CHOICES)
        self.conv1 = nn.Conv2d(self.INPUT_CHANNEL, 9, 3)
        self.bn1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, self.MID_CHANNEL, 3)
        self.bn2 = nn.BatchNorm2d(self.MID_CHANNEL)
        # self.conv3 = nn.Conv2d(27, self.MID_CHANNEL, 3)
        # self.bn3 = nn.BatchNorm2d(self.MID_CHANNEL)
        self.fc = nn.Linear(
            (self.INPUT_HEIGHT-4)*(self.INPUT_WIDTH-4)*self.MID_CHANNEL,
            self.MID_CHANNEL
        )
        self.head = nn.Linear(self.MID_CHANNEL, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.head(x)
        return x

    def choice(self, action):
        return self.CHOICES[action]

    @property
    def input_size(self):
        return [1, self.INPUT_CHANNEL, self.INPUT_HEIGHT, self.INPUT_WIDTH]

#TODO: Modelと変数名などを合わせる
class CustomModel(nn.Module):
    def __init__(self, choices,
            conv_channels=[9, 16], linear_outputs=[], kernel_size=3,
            input_height=240, input_width=320, input_channels=3):
        super(CustomModel, self).__init__()
        self.channel = input_channels
        self.height = input_height
        self.width = input_width
        self.choices = choices

        conv_layers = []
        linear_layers = []
        # conv_channels = [input_channels] + conv_channels
        in_channel = input_channels
        for out_channel in conv_channels:
            conv_layers += [
                nn.Conv2d(in_channel, out_channel, kernel_size),
                nn.BatchNorm2d(out_channel),
            ]
            in_channel = out_channel
        _linear_outputs = linear_outputs + [len(choices)]
        num_conv = len(conv_channels)
        size_reduction = num_conv * (kernel_size - 1)
        in_size = (input_height - size_reduction) * (input_width - size_reduction) * in_channel
        for out_size in _linear_outputs:
            linear_layers.append(nn.Linear(in_size, out_size))
            in_size = out_size
        self.conv = nn.Sequential(*conv_layers)
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def choice(self, action):
        return self.choices[action]

    @property
    def CHOICES(self):
        return self.choices

    @property
    def input_size(self):
        return [1, self.channel, self.height, self.width]
