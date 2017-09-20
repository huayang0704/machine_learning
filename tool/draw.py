#!/usr/bin/python
#coding=utf8

import sys
import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


reload(sys)
sys.setdefaultencoding("utf-8")

##************************************************************************
## ** 画图脚本
## **
## **
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************


def d1():
	x=[1,2,3,4,5]
	y=[5,7,3,8,1]
	plt.plot(x,y)
	plt.show()


def d2():
	x1 = [1,2,3,4,5]
	y1 = [5,7,3,8,1]
	x2 = [1,2,3,4,5]
	y2 = [10,11,7,7,3]

	plt.title('2 dimensional dataset')
	plt.plot(x1, y1,'b',lw=1.5, label='First Line')
	plt.plot(x2, y2,'r',lw=1.5, label='Second Line')
	plt.xlabel('index')
	plt.ylabel('value')
	plt.legend() #加图例
	plt.show()

def d3():
	x = np.linspace(0,1,30,endpoint=True)
	y = x**2 - 2*x
	#plt.ylim(0,10)

	y1 = -x**2 + 2*x
	#plt.ylim(-10,0)

	plt.plot(x,y)
	plt.plot(x,y1)

	plt.show()
	

def d4():
	print ''	

if __name__ == "__main__":
	start = datetime.datetime.now()
		
	#d1()
	#d2()
	d3()

	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
