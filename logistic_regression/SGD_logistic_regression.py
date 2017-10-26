# coding: utf-8
# author: Ryan
'''
A simple Logistic Regression model.
'''
from __future__ import division
import sys
import os
import random
import math
import collections
import array
import operator


class SimpleLogisticRegression(object):
    def __init__(self, alpha, feature_num):
        """构造函数

        参数
        ---
        alpha: double
            学习率

        feature_num: int
            特征数量
        """
        self.__alpha = alpha
        self.__features_num = feature_num
        self.__coef = [0.] * self.__features_num
        self.__intercept = 0.

    def fit(self, X, y, verbose=False):
        """训练模型。

        返回值
        ------
            模型训练的最终log似然值。

        随机梯度下降算法
        """
        last_target = -sys.maxint
        last_step = 0
        step = 0
        while True:
            step += 1
            gradient = [0.] * (self.__features_num + 1)
            for tx, ty in zip(X, y):
                delta = ty - self.__sigmoid(tx)

                # 方式一
                #for i, xi in enumerate(tx):
                #    gradient[i] = delta * xi
                #gradient[-1] = delta
                #self.__coef = map(lambda c, g: c + self.__alpha * g, self.__coef, gradient[:-1])
                #self.__intercept += self.__alpha * gradient[-1]

                # 方式二
                self.__coef = map(lambda c, g: c + self.__alpha * delta * g, self.__coef, tx)
                self.__intercept += self.__alpha * delta

            # gradient = map(lambda g: g / len(X), gradient)


            target = sum(map(lambda py, ty: ty * math.log(py) + (1 - ty) * math.log(1 - py),
                             map(self.__sigmoid, X), y)) / len(X)
            if target - last_target > 1e-8:
                last_target = target
                last_step = step
            elif step - last_step >= 10:
                break

            if verbose is True and step % 1000 == 0:
                sys.stderr.write("step %s: %.6f\n" % (step, target))

        target = sum(map(lambda py, ty: ty * math.log(py) + (1 - ty) * math.log(1 - py),
                         map(self.__sigmoid, X), y)) / len(X)
        if verbose is True:
            sys.stderr.write("Final training error: %.6f\n" % (target))
        return target

    def predict(self, X):
        """输出每个样本的预测值。

        返回值
        -----
            列表类型，长度与X的样本量相同。
        """
        if not self._check_columns(X):
            sys.stderr.write("The data to be evaluated can't match training data's features\n")
            return None
        return map(self.__sigmoid, X)

    def __sigmoid(self, x):
        """sigmoid函数，返回sigmoid函数值"""
        return 1. / (1 + math.exp(-sum(map(operator.mul, self.__coef, x)) - self.__intercept))

    def _check_columns(self, X):
        """检查每个样本的类型是否是数组或者列表类型，并且长度与特征数相等。

        返回值
        -----
            数据合法时，返回True，否则返回False。
        """
        for x in X:
            if not isinstance(x, (list, tuple, array.array)):
                return False
            if len(x) != self.__features_num:
                return False
        return True


if __name__ == "__main__":
    lr = SimpleLogisticRegression(0.1, 3)
    X = [[1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8]]
    y = [0, 0, 1, 1]
    print lr.predict(X)
    lr.fit(X, y, verbose=True)
    print lr.predict(X)
