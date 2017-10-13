#!/usr/bin/python
#coding=utf8

##************************************************************************
## ** pandas使用示例
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
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from perceptron.perceptron import *
from common.gradient import *

reload(sys)
sys.setdefaultencoding("utf-8")



if __name__ == "__main__":
	start = datetime.datetime.now()
	#创建一个series对象,pandas默认会创建整数索引
	s = pd.Series([1,3,5,np.nan,6,8])
	print s

	#创建一个DataFrame
	dates = pd.date_range('20130101', periods=6)
	print dates
	#以日期为索引,A,B,C,D为列名,6行4列的随机数据(一张表)
	df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
	print df

	#通过字典来构建DataFrame
	df2 = pd.DataFrame({
		'A':1,
		'B':pd.Timestamp('20130101'), 
		'C':pd.Series(1,index=list(range(4)), dtype='float32'),
		'D':np.array([3]*4, dtype='int32'),
		'E':pd.Categorical(['test', 'train', 'test', 'train']),
		'F':'foo'
	})
	print df2
	print df2.dtypes

	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
