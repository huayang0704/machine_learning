#!/usr/bin/python
# coding=utf8

##************************************************************************
## ** 二分类模型评价指标AUC计算
## **
## **create: 2017-10-24
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************


import sys
import os
import datetime
import numpy as np
from sklearn.metrics import roc_curve, auc


# sys.path.append('../')
#from perceptron.perceptron import *
#from common.gradient import *

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    start = datetime.datetime.now()

    y = np.array([1, 1, 2, 2])
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
    print(fpr)
    print(tpr)
    print(thresholds)
    print(auc(fpr, tpr))

    end = datetime.datetime.now()
    print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
