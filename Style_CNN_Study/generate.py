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

'''
这部分的变量后期要认真关注
'''
###############################################################################
# Algorithm constants
###############################################################################
# 设置随机噪声图像与内容图像的比率
NOISE_RATIO = 0.6
# 设置迭代次数
ITERATIONS = 1000
# 设置内容图像与风格图像的权重
alpha = 1
beta = 500
# 加载VGG-19 MODEL及设定均值
VGG_Model = 'MODEL/imagenet-vgg-verydeep-19.mat'
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
# 设置需要用到的卷积层
CONTENT_LAYERS = [('conv4_2', 1.)]
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]

#生成随机噪声图，与content图以一定比率融合一起
def generate_noise_image(content_image,noise_ratio = NOISE_RATIO):
	noise_image = np.random.unifom(
	-20,20,
	(1,IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_CHANNELS)).astype('float32')
	# 将生成的白噪声叠加到内容content图片上，
	# 风格将叠加在该函数生成的图片上
	img = noise_image * noise_ratio + content_image * (1 - noise_ratio)
	return img

#加载图片并声称向量	
def load_image(path):
	image = scipy.misc.imread(path)
	# Resize the image for convnet input, there is no change but just
	# add an extra dimension.
	image = np.reshape(image, ((1,) + image.shape))
	# Input to the VGG net expects the mean to be subtracted.
	image = image - MEAN_VALUES
	return image

#保存图片到path
def save_image(path,image):
	# Output should add back the mean.
	image = image + MEAN_VALUES
	# Get rid of the first useless dimension, what remains is the image.
	image = image[0]
	image = np.clip(image, 0, 255).astype('uint8')
	scipy.misc.imsave(path, image)

# 生成一层CNN神经网络(可选择是卷积层还是池化pool层)
# ntype:'conv'或者'pool'
# nin:	输入的神经网络层tensor
# nwb:	该层神经网络层的w和b	(是一个数组，nwb[0]是w,nwb[1]是b)
def build_net(ntype,nin,nwb=None):
	if ntype == 'conv':
	# 卷积层:以步长为1，初始化W为nwb[0],偏移b为nwb[1],
	# padding模式是SAME
		return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME')+nwb[1])
	elif ntype == 'pool':
	# 池化层:以步长为2，padding模式是SAME
		return tf.nn.avg_pool(nin,ksize=[1,2,2,1],
								strides=[1,2,2,1],padding='SAME')

# 获取VGG-19模型已经训练好的CNN模型的W,b的初始值，
# 以加速训练过程								
def get_weight_bias(vgg_layers, i):
	weights = vgg_layers[i][0][0][2][0][0]
	weights = tf.constant(weights)
	bias = vgg_layers[i][0][0][2][0][1]
	bias = tf.constant(np.reshape(bias, (bias.size)))
	return weights, bias
	
# 构建 VGG19 模型	
def build_vgg19(path):
	
	pass
	
# content内容的 单层神经网络的loss
def content_layer_loss(p,x):
	
	pass
	
# 总的content_image的loss
def content_loss_func(sess,net):
	
	pass
	

def gram_matrix(x,area,depth):
	
	pass
	
# style风格的 单层神经网络的loss	
def style_layer_loss(a,x):
	
	pass
	
# 总的style_image的loss
def style_loss_func(sess,net):
	
	pass
	
# 主函数
def main():
	
	pass
	
	
#相当于该python代码的主程序执行入口
if __name__ == '__main__':
	main()


