#!/usr/bin/python
#coding=utf8

import tensorflow as tf
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding( "utf-8" )

# 生产100个随机数
x_data = np.random.rand(100).astype(np.float32)
print x_data
#print len(x_data)
y_data = 0.1 * x_data + 0.3

# 开始创建tensorflow结构
Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
print 'Weight:', Weight
biases = tf.Variable(tf.zeros([1]))
print 'biases:', biases
y = Weight * x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# 结束创建tensorflow结构

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print (step, sess.run(Weight), sess.run(biases))
