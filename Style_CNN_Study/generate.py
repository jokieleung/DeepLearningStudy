# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

'''
generate.py
用于训练并生成风格化的主程序
尝试将学习的图像风格化代码跑在美团云上
'''

'''
全局变量的定义
'''
# Output folder for the images.
OUTPUT_DIR = 'output/'
# Style image to use.
STYLE_IMAGE = '/images/ocean.jpg'  #到时更改为我的测试风格图片
# Content image to use.
CONTENT_IMAGE = '/images/Taipei101.jpg'#到时更改为我的内容图片
# Image dimensions constants.
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3


#生成随机噪声图，与content图以一定比率融合一起
def generate_noise_image(content_image,noise_ratio = NOISE_RATIO):
	
	pass


