#!/usr/bin/python
#coding=utf8

import os
import sys
from perceptron import Perceptron


reload(sys)
sys.setdefaultencoding("utf-8")


#定义激活函数
f = lambda x:x

class LinearUnit(Perceptron):
	def __init__(self, input_num):
		'''
		初始化线性单元,设置输入参数的个数
		'''
		Perceptron.__init__(self, input_num, f)

	
def train_linear_unit():
	'''
	使用数据训练线性单元
	'''
	lu = LinearUnit(1)
	input_vecs, labels = get_training_dataset()
	lu.train(input_vecs, labels, 20, 0.01)

	return lu

def get_training_dataset():
	'''

	'''
	# 输入向量列表，每一项是工作年限
	input_vecs = [[5], [3], [8], [1.4], [10.1]]
	# 期望的输出列表，月薪，注意要与输入一一对应
	labels = [5500, 2300, 7600, 1800, 11400]

	return input_vecs, labels 

if __name__ == "__main__":
	'''
	训练线性单元
	'''
	linear_unit = train_linear_unit()
	print linear_unit
	#测试
	print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
	print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
	print 'Work 2 years, monthly salary = %.2f' % linear_unit.predict([2])
	print 'Work 20 years, monthly salary = %.2f' % linear_unit.predict([20])
