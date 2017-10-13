#!/usr/bin/python
#coding=utf8

##************************************************************************
## ** 模板脚本
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
from perceptron.perceptron import *
from common.gradient import *

reload(sys)
sys.setdefaultencoding("utf-8")



if __name__ == "__main__":
	start = datetime.datetime.now()


	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
