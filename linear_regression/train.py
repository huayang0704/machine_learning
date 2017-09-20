#!/use/bin/python
#coding=utf8

import paddle.v2 as paddle
import sys
import os
from PIL import Image
import numpy as np
import paddle.v2.dataset.uci_housing as uci_housing

reload(sys)
sys.setdefaultencoding( "utf-8" )

def main():
	# init
	paddle.init(use_gpu=False, trainer_count=1)
	
	# network config
	# name:定义一个名字,type:一维数据的个数
	x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
	y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))
	#预测的结果,size:,act:激活函数为线性回归
	y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
	#损失函数为均方误差
	cost = paddle.layer.square_error_cost(input=y_predict, label=y)
	
	# create parameters
	parameters = paddle.parameters.create(cost)
	
	# create optimizer
	optimizer = paddle.optimizer.Momentum(momentum=0)
	
	trainer = paddle.trainer.SGD(
	cost=cost, parameters=parameters, update_equation=optimizer)
	
	#定义输出信息的dict
	feeding = {'x': 0, 'y': 1}
	
	# event_handler to print training and testing info
	def event_handler(event):
		if isinstance(event, paddle.event.EndIteration):
			if event.batch_id % 100 == 0:
				print "Pass %d, Batch %d, Cost %f" % ( event.pass_id, event.batch_id, event.cost)

		if isinstance(event, paddle.event.EndPass):
			if event.pass_id % 10 == 0:
				with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
					parameters.to_tar(f)
			result = trainer.test( reader=paddle.batch(uci_housing.test(), batch_size=2), feeding=feeding)
			print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # training
	trainer.train(reader=paddle.batch(paddle.reader.shuffle(uci_housing.train(), buf_size=500), batch_size=2), 
				  feeding=feeding, 
				  event_handler=event_handler, 
				  num_passes=30)

    # inference
	test_data_creator = paddle.dataset.uci_housing.test()
	test_data = []
	test_label = []
	for item in test_data_creator():
		test_data.append((item[0], ))
		print '=============:', item[0]
		test_label.append(item[1])
		if len(test_data) == 5:
			break

	print 'test_data:', type(test_data)
	print 'test_data:', test_data

    # load parameters from tar file.
    # users can remove the comments and change the model name
    # with open('params_pass_20.tar', 'r') as f:
    #     parameters = paddle.parameters.Parameters.from_tar(f)

	probs = paddle.infer(output_layer=y_predict, parameters=parameters, input=test_data)

	print probs

	for i in xrange(len(probs)):
		print "label=" + str(test_label[i][0]) + ", predict=" + str(probs[i][0])

	print type(parameters)
	print 'parameters:', parameters

if __name__ == '__main__':
	main()
