#!/usr/bin/env python
import rospy
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import numpy as np
import time
from PIL import Image as IMG

import cv2
from cv_bridge import CvBridge

from model import CustomModel

model = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def init_inference(model_kwargs):
    global model
    global device
    model = CustomModel(**model_kwargs)
    model.eval()

    if args.trt_module :
        from torch2trt import TRTModule
        if args.trt_conversion :
            model.load_state_dict(torch.load(args.pretrained_model))
            model = model.cuda()
            x = torch.ones((1, 3, 240, 320)).cuda()
            from torch2trt import torch2trt
            model_trt = torch2trt(model, [x], max_batch_size=100, fp16_mode=True)
            #model_trt = torch2trt(model, [x], max_batch_size=100)
            torch.save(model_trt.state_dict(), args.trt_model)
            exit()
        model_trt = TRTModule()
        #model_trt.load_state_dict(torch.load('road_following_model_trt_half.pth'))
        model_trt.load_state_dict(torch.load(args.trt_model))
        model = model_trt.to(device)
    else :
        model.load_state_dict(torch.load(args.pretrained_model))
        model = model.to(device)


i=0
pre = time.time()
now = time.time()
# tmp = torch.zeros([1,3,120,320])
tmp = torch.zeros([1,3,240,320])
bridge = CvBridge()
twist = Twist()
choices = [-0.5, 0.0, 0.5]

def set_throttle_steer(data):
    global i
    global pre
    global now
    global tmp
    global bridge
    global twist
    global model
    global device
    global choices

    i=i+1
    if i == 100 :
        pre = now
        now = time.time()
        i = 0
        print ("average_time:{0}".format((now - pre)/100) + "[sec]")
    start = time.time()
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = torch.from_numpy(np.copy(image).astype(np.float32)).view(1, 3, 240, 320)
    # image = IMG.fromarray(image)
    # image = transforms.ToTensor()(image)
    # image = transforms.ToTensor()(image)[:, :120, :]
    
    # tmp[0] = image
    
    #tmp = tmp.half()
    image = image.to(device)
    # image = tmp.to(device)
    image = image[:, :, 120:, :]
    if i < 5:
        torchvision.utils.save_image(image, "/home/jetson/save_dir/inference_image_2_{}.png".format(i))
    # image = tmp.to(device)
    outputs = model(image)

    # output = choices[model(image).max(1)[1].view(1, 1)].cpu()
    outputs_np = outputs.to('cpu').detach().numpy().copy()
    print(outputs_np)
    
    # output = np.argmax(outputs_np, axis=1)
    # output = choices[np.argmax(outputs_np, axis=1)]
    _output = np.argmax(outputs_np, axis=1)
    print(_output)
    output = choices[int(_output)]
    print(output)
    
    #angular_z = (float(output)-256)/100
    # angular_z = (float(output)-1)
    angular_z = output
    twist.linear.x = 1.6
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = -angular_z
    twist_pub.publish(twist)
    end = time.time()
    print ("time_each:{0:.3f}".format((end - start)) + "[sec]")


def inference_from_image():
    global twist_pub
    rospy.init_node('inference_from_image', anonymous=True)
    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    image_topic_name = args.image_topic_name
    rospy.Subscriber(image_topic_name, Image, set_throttle_steer)
    r = rospy.Rate(10)
    #while not rospy.is_shutdown():
    #    r.sleep()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Autonomous with inference")
	
	arg_parser.add_argument("--trt_conversion", action='store_true')
	arg_parser.add_argument("--trt_module", action='store_true')
	arg_parser.add_argument("--pretrained_model", type=str, default='/home/jetson/save_dir/model.pth')
	arg_parser.add_argument("--trt_model", type=str, default='road_following_model_trt.pth' )
	arg_parser.add_argument("--image_topic_name", type=str, default='/front_camera/image_raw' )

	args = arg_parser.parse_args()

	return args

if __name__ == '__main__':
    args = parse_args()
    model_kwargs={
        "conv_channels": [16, 32, 32],
        "kernel_size": 5,
        "stride": 2,
        "choices": choices,
        "input_height": 120,
        "input_width": 320,
        "input_channels": 3,
    }
    init_inference(model_kwargs)
    try:
        inference_from_image()
    except rospy.ROSInterruptException:
        pass