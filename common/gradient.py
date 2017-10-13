#!/usr/bin/python
#coding=utf8

##************************************************************************
## ** 随机梯度下降算法公共脚本
## ** 参数迭代规则: 
## ** 	w = w + n * (t - y) * x, w权重向量,n学习率,t标签值,y预测值,x输入参数
## ** 
## **
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************

import os
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


class Gradient(object):
	def __init__(self, input_num, activator):
		'''
		初始化感知器,设置输出参数的个数,以及激活函数
		input_num: 参数(权重w)的个数
		activator: 激活函数
		'''
		self.activator = activator
		#权重初始化为0
		self.weights = [0.0 for _ in range(input_num)]
		#偏置项初始化为0
		self.bias = 0.0

	def __str__(self):
		'''
		打印学习到的权重和偏置项
		'''
		return 'weights:\t%s\nbias:\t%f\n' % (self.weights, self.bias)

	def predict(self, input_vec):
		#输入向量,输出感知器的计算结果
		#total = 0.0
		#for (vec, weight) in zip(input_vec, self.weights):
		#	total += vec * weight	
		#return self.activator(total + self.bias)
		return self.activator(reduce(lambda a,b: a + b, map(lambda (x,w): x * w, zip(input_vec, self.weights)) , 0.0) + self.bias)	

	def train(self, input_vecs, labels, iteration, rate):
		'''
		输入训练数据
		input_vecs: 一组向量、
		labels: 与每个向量对应的label
		interation: 以及训练的轮数
		rate: 学习率
		'''
		for i in range(iteration):
			self._one_iteration(input_vecs, labels, rate)

	def _one_iteration(self, input_vecs, labels, rate):
		'''
		一次迭代把所有的数据都过一编
		'''
		# 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
		# 而每个训练样本是(input_vec, label)
		samples = zip(input_vecs, labels)
		#对每个样本按照感知器规则更新权重
		for(input_vec, label) in samples:
			#计算感知器在当前权重下的输出
			output = self.predict(input_vec)
			#更新权重
			self._update_weights(input_vec, output, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		'''
		按照感知器规则更新权重
		'''
		# 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
		# 变成[(x1,w1),(x2,w2),(x3,w3),...]
		# 然后利用感知器规则更新权重
		#print 'imput_vec:', input_vec,'output:',output
		#print 'befor bias:', self.bias, 'befor weights:', self.weights
		delta = label - output
		self.weights = map(lambda (x,w): w + rate * delta * x, zip(input_vec, self.weights))
		#更新bias
		self.bias += rate * delta
		#print 'training:**********'
		print 'bias:', self.bias, 'weights:', self.weights
		print '\n'

###--------------------------------------------------------------------------------------------

def f(x):
	'''
	激活函数
	'''
	return 1 if x > 0 else 0


def get_training_dataset():
	'''
	基于and真值表构建训练数据
	'''
	# 构建训练数据
	# 输入向量列表
	input_vecs = [[1,1], [1,0], [0,1], [0,0]]
	labels = [1,0,0,0]

	return input_vecs, labels

def get_training_dataset2():
	'''
	基于异或真值表构建训练数据
	'''
	# 构建训练数据
	# 输入向量列表
	input_vecs = [[1,1], [1,0], [0,1], [0,0]]
	labels = [0,1,1,0]

	return input_vecs, labels


def train_and_perceptron():
	'''
	使用and真值表训练感知器
	'''
	# 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
	p = Gradient(2, f)
	# 训练，迭代10轮, 学习速率为0.1
	input_vecs, labels = get_training_dataset()
	#input_vecs, labels = get_training_dataset2()
	p.train(input_vecs, labels, 5, 0.1)
	#返回训练好的感知器

	return p

if __name__=="__main__":
	# 训练and感知器
	and_perception = train_and_perceptron()
	# 打印训练获得的权重
	print and_perception
	# 测试
	print '1 and 1 = %d' % and_perception.predict([1, 1])
	print '0 and 0 = %d' % and_perception.predict([0, 0])
	print '1 and 0 = %d' % and_perception.predict([1, 0])
	print '0 and 1 = %d' % and_perception.predict([0, 1])

