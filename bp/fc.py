#!/usr/bin/python
#coding=utf8

##************************************************************************
## ** 全链接实现
## **
## **
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************


import sys
import os
import datetime
import numpy as np

from perceptron.perceptron import *
from common.ml import *

reload(sys)
sys.setdefaultencoding("utf-8")


class FullConnectedLayer(object):
	def __init__(self, input_size, output_size, activator):
		'''
		input_size:本层输入向量的维度
		output_size:本层输出向量的维度
		activator:激活函数
		'''
		self.input_size = input_size
		self.output_size = output_size
		self.activator = activator
		#权重数组
		self.W = np.random.uniform(-0.1, 0.1, (input_size, output_size))
		#偏置项b
		self.b = np.zeros((input_size, 1))
		#输出向量
		self.output = np.zeros((output_size, 1))

	def forward(self, input_array):
		'''
		前向计算
		input_array:输入向量,维度必须和input_size相等
		'''
		self.input = input_array
		self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

	def backward(self, delta_array):
		'''
		反向计算W和b梯度
		dwlta_array:从上一层传递过来的误差项
		'''
		self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
		self.W_grad = np.dot(delta_array, self.input.T)
		self.b_grad = delta_array

	def update(self, learning_rate):
		'''
		使用梯度下降算法更新权重
		'''
		self.W += learning_rate * self.W_grad
		self.b += learning_rate * self.b_grad

#神经网络类
class Network(object):
	def __init__(self,layers):
		self.layers = []
		for i in range(len(layers) - 1):
			self.layers.append(FullConnectedLayer(layers[i], layers[i+1]), SigmoidActivator())

	def predict(self, sample):
		'''
		sample:输入样本
		'''
		output = sample
		for layer in self.layers:
			layer.forward(output)
			output = layer.output

		return output

	def train(self, labels, data_set, rate, epoch):
		'''
		训练函数
		labels:样本标签
		data_set:输入样本
		rate:学习率
		epoch:训练的轮数
		'''
		for i in range(epoch):
			for d in range(len(data_set)):
				self.train_one_sample(labels[d], data_set[d], rate)

	def train_one_sample(self, label, sample, rate):
		self.predict(sample)
		self.calc_gradient(label)
		self.update_weight(rate)

	def calc_gradient(self, label):
		delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
		for layer in self.layers[::-1]:
			layer.backward(delta)
			delta = layer.delta

		return delta

	def update_weight(self, rate):
		for layer in self.layers:
			layer.update(rate)


def transpose(args):
	return map(
		lambda arg:map(
			lambda line:np.array(line).reshape(len(line), 1), arg	
		), args
	)


if __name__ == "__main__":
	start = datetime.datetime.now()


	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
