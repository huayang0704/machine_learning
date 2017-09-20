#!/usr/bin/python
#coding=utf8

import sys
import os
import datetime
from  numpy import *


reload(sys)
sys.setdefaultencoding("utf-8")


def sigmoid(inx):
	return 1.0 / (1 + exp(-inx))

#节点类,负责记录和维护节点自身信息以及节点相关的上下游链接,实现输出值和误差项的计算
class Node(object):
	def __init__(self, layer_index, node_index):
		'''
		构造节点编号
		layer_index:节点所属层的编号
		node_index:节点的编号
		'''
		self.layer_index = layer_index
		self.node_index = node_index
		self.downstream = []
		self.upstream = []
		self.output = 0
		self.delta = 0

	def set_output(self, output):
		'''
		设置节点的输出值,如果节点输入输入层会用到这个函数
		'''
		self.output = output

	def append_downstream_connection(self, conn):
		'''
		添加一个下游节点的链接
		'''
		self.downstream.append(conn)

	def append_upstream_connection(self, conn):
		'''
		添加一个上游节点的链接
		'''
		self.upstream.append(conn)

	def calc_output(self):
		'''
		根据 y = sigmoid(w * x), w,x向量
		'''
		output = reduce(lambda ret,conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)	
		self.output = sigmoid(output)

	def clac_hidden_layer_delta(self):
		'''
		节点属于隐藏层,计算detal
		'''


if __name__ == "__main__":
	start = datetime.datetime.now()


	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
