#!/usr/bin/env python
#coding=utf8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import pandas as pd
import _0_config
import time
import json


print ("Local current time :", time.asctime( time.localtime(time.time()) ))

#================
# 训练数据
#================
dftrain = pd.read_json(path_or_buf=_0_config.trainDataPath)
trainPrice = dftrain['price'].as_matrix()
dftrain = dftrain.drop(['price'], 1, inplace=False)

#================
# 测试数据
#================
dftest = pd.read_json(path_or_buf=_0_config.testDataPath)
testPrice = dftest['price'].as_matrix()
dftest = dftest.drop(['price'], 1, inplace=False)

#================
# 评估数据
#================
dfpredict = pd.read_json(path_or_buf=_0_config.predictDataPath);


#================
# 特征
#================
features = []
for column in dftrain.columns:
    features.append(tf.contrib.layers.real_valued_column(column, dimension=1))

#================
# 参数配置
#opt = tf.train.Optimizer(learning_rate=100)
# Set model params
#0.0000005      NaN
#0.00000025     NaN
#0.0000002            loss: 1743.8076

#0.00000015           loss: 1990.4282

#0.0000001            loss: 1732.9126       1593.0297
#0.00000005     50m   loss: 1965
#0.000000005    50m   loss: 2179.1982
#0.0000000025   50m   loss: 2203
#0.0000000005   50m+  loss: 2221
#0.00000000025  50m   loss: 2314
#0.00000000005  50m+  loss: 5150+

# remove some features
#0.00000005  2075.9253
#0.000000025 2327.4172
#0.00000009  1700.3657
#0.0000001   1673.3259
#0.000000105 1563.4338
#0.000000109 1698.0518
#0.00000013  1844.4138
#0.00000015  2066.5925
#0.0000002   3598.0186
#================

#opt = tf.train.GradientDescentOptimizer(learning_rate=0.000000109)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.0000000109)
#===线性回归
estimator = tf.contrib.learn.LinearRegressor(
        feature_columns=features,
        optimizer=opt,
        #model_dir="./model/"
    )

#===神经网络回归
'''
estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=features,
        hidden_units=[50, 20, 50, 10],
        activation_fn=tf.nn.relu,
        optimizer=opt,
        #model_dir="./saved_model/"
        )
'''

xs_train = {}
xs_eval = {}
xs_predict = {}
try:
for column in dftrain.columns:
    xs_train[column] = dftrain[column].as_matrix()
    xs_eval[column] = dftest[column].as_matrix()
    xs_predict[column] = dfpredict[column].as_matrix()
except:
	print 'exception'

y_train = trainPrice
y_eval = testPrice


input_fn = tf.contrib.learn.io.numpy_input_fn(
    xs_train,
    y_train,
    batch_size=1000,
    num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    xs_eval,
    y_eval,
    batch_size=1000,
    num_epochs=1000)

print('start fit')
estimator.fit(input_fn=input_fn, steps=10000)
print('end fit')
print('end train loss')

#train_loss = estimator.evaluate(input_fn=input_fn)

#eval_loss = estimator.evaluate(input_fn=eval_input_fn)
#print("train loss: %r"% train_loss)
#print("eval loss: %r"% eval_loss)
print ("Local current time :", time.asctime( time.localtime(time.time()) ))


#================
# 评估在售房价
#================
predictResult = list(estimator.predict(xs_predict))
df = pd.DataFrame({'price': predictResult})
df = df.reset_index()
df.to_json(path_or_buf=_0_config.resultDataPath)
#print("predictL%r"% predictResult)

exit

#=========

# 判断是否有nan
#print(np.any(np.isnan(xs_train[column])))
#======
