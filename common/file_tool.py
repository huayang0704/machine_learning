#!/usr/bin/python
#coding=utf8

##************************************************************************
## ** 文件处理
## **
## **create: 2017-10-13
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************


import sys
import os
import datetime
import numpy as np

#sys.path.append('../')
#from perceptron.perceptron import *
#from common.gradient import *

reload(sys)
sys.setdefaultencoding("utf-8")


def read_file(file_path):
	'''
    按行读取文本
    :param file_path:
    :return:按行,数组
    '''
	datas = list()
	with open(file_path, 'r') as f:
		datas = f.readlines()

	# try:
	# 	file = open(file_path)
	# 	while True:
	# 		lines = file.readlines(100000)
	# 		datas.extend(lines)
	# 		if not lines:
	# 			break
	# finally:
	# 	if file:
	# 		file.close()

	return datas

def read_file_list(file_path, delimiter=","):
	'''
    按行读取文本
    :param file_path:
    :return:数组
    '''
	with open(file_path, 'r') as f:
		datas = f.readline()

	return datas.split(delimiter)

def write_list(file_path, datas):
	'''
	写文件
	:param file_path:
	:param datas: 数组
	:return:
	'''
	with open(file_path, 'w+') as f:
		f.writelines(",".join(datas))

def write(file_path, datas):
	'''
	写文件
	:param file_path:
	:param datas: 字符串
	:return:
	'''
	with open(file_path, 'w+') as f:
		f.write(datas)

#---------------------------------------
###
# numpy读取文件
###
def save_parameters(file_name, datas):
	'''
	:param file_name:
	:param datas: numpy数组
	:return:
	'''
	np.savetxt(file_name, datas, fmt="%.3f", delimiter=" ")

def load_parameters(file_name):
	'''
	:param file_name:
	:return:返回numpy数组
	'''
	return np.loadtxt(file_name)

if __name__ == "__main__":
	start = datetime.datetime.now()

	# write_list("test.txt", ['1', '2'])
	# a = np.array([[1, 2], [3, 4]])
	a = np.array([1, 2, 3, 4])
	file_name = "a.txt"
	save_parameters(file_name, a)
	print load_parameters(file_name)

	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
