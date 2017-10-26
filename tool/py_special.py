#!/usr/bin/python
# coding=utf8

import sys
import operator
import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding("utf-8")

##************************************************************************
## ** python特殊语法
## **
## **
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************

def pass_test():
    pass


def filter_f(p):
    return p if p != 's' else None


def map_f(p):
    return p * 2


def reduce_f(x, y):
    return x + y


def special():
    # pass_test()
    '''
    对sequence中的item依次执行function,将结果为True的item组成一个新的sequence返回
    '''
    print filter(filter_f, ['a', 'b', 's', 'c'])
    print filter(lambda x: x if x != 's' else None, ['a', 'b', 's', 'c'])

    '''
    对sequence的每一个item依次做map_f(p)操作
    '''
    print map(map_f, ['a', 'b', 'c', 2])
    print map(lambda x: x * 2, ['a', 'b', 'c', 2])
    print map(operator.mul, [1, 2, 3], [1, 2, 3])

    '''
    对sequence的每一个item依次与上一次的结果做reduce_f(x,y)操作
    '''
    print reduce(reduce_f, [1, 2, 3, 4])
    print reduce(reduce_f, [1, 2, 3, 4], 10)
    print reduce(lambda x, y: x + y, [1, 2, 3, 4], 10)
    print reduce(lambda x, y: x + y, ['a', 'b', 'c'])


def outter():
    x = 1
    print x

    def inner():
        print x

    return inner


def p(x, y, *args):
    print x, y, args


if __name__ == "__main__":
    start = datetime.datetime.now()

    p(1,2,3,4,5)

    # special()

    # foo = outter()
    # foo()

    end = datetime.datetime.now()
    print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
