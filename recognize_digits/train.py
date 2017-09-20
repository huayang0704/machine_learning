#!/use/bin/python
#coding=utf8

import paddle.v2 as paddle
import sys
import os
from PIL import Image
import numpy as np
import datetime

reload(sys)
sys.setdefaultencoding( "utf-8" )

# # 定义三种分类器

#1,Softmax回归:只通过一层简单的以softmax为激活函数的全连接层，就可以得到分类的结果
def softmax_regression(img):
	#size:表示输出的大小
	predict = paddle.layer.fc(input=img, size=10, act=paddle.activation.Softmax())	

	return predict

#多层感知器
def multilayer_perceptron(img):
	# 第一个全连接层，激活函数为ReLU
	hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
	# 第二个全连接层，激活函数为ReLU
	hidden2 = paddle.layer.fc(input=hidden1, size=64, act=paddle.activation.Relu())
	# 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
	predict = paddle.layer.fc(input=hidden2, size=10, act=paddle.activation.Softmax())

	return predict

#卷积神经网络
def convolutional_neural_network(img):
    # first conv layer
	conv_pool_1 = paddle.networks.simple_img_conv_pool(
		input=img,
		filter_size=5,
		num_filters=20,
		num_channel=1,
		pool_size=2,
		pool_stride=2,
		act=paddle.activation.Relu())
	# second conv layer
	conv_pool_2 = paddle.networks.simple_img_conv_pool(
		input=conv_pool_1,
		filter_size=5,
		num_filters=50,
		num_channel=20,
		pool_size=2,
		pool_stride=2,
		act=paddle.activation.Relu())
	# fully-connected layer
	predict = paddle.layer.fc(input=conv_pool_2, size=10, act=paddle.activation.Softmax())

	return predict

def main():
	paddle.init(use_gpu=False, trainer_count=1)
	#define network topology
	images = paddle.layer.data(name='pixel', type=paddle.data_type.dense_vector(784))
	#print type(images)
	#print images
	label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(10))
	#print type(label)
	#print label

	#predict = softmax_regression(images)
	predict = multilayer_perceptron(images)
	#predict = convolutional_neural_network(images)

	cost = paddle.layer.classification_cost(input=predict, label=label)
	parameters = paddle.parameters.create(cost)

	optimizer = paddle.optimizer.Momentum(learning_rate=0.1 / 128.0, momentum=0.9, regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128)) 
	trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

	lists = []

	def event_handler(event):
		#if isinstance(event, paddle.event.EndIteration):
		#	if event.batch_id % 100 == 0:
		#		print "Pass %d, Batch %d, Cost %f, %s" % ( event.pass_id, event.batch_id, event.cost, event.metrics)
		if isinstance(event, paddle.event.EndPass):
			# save parameters
			with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
				parameters.to_tar(f)

		result = trainer.test(reader=paddle.batch( paddle.dataset.mnist.test(), batch_size=128))
		#result = trainer.test(reader=paddle.batch(paddle.dataset.mnist.test(), batch_size=1))
		#print "Test with Pass %d, Cost %f, %s\n" % (event.pass_id, result.cost, result.metrics)
		lists.append((event.pass_id, result.cost, result.metrics['classification_error_evaluator']))

	#trainer.train(
	#	reader=paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192), batch_size=128), 
	#	event_handler=event_handler,
	#	num_passes=5)
	trainer.train(
		reader=paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=16), batch_size=2), 
		event_handler=event_handler,
		num_passes=1)

	# find the best pass
	best = sorted(lists, key=lambda list: float(list[1]))[0]
	print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
	print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)
	
	def load_image(file):
		im = Image.open(file).convert('L')
		im = im.resize((28, 28), Image.ANTIALIAS)
		im = np.array(im).astype(np.float32).flatten()
		im = im / 255.0
		return im

	test_data = []
	cur_dir = os.path.dirname(os.path.realpath(__file__))
	test_data.append((load_image(cur_dir + '/infer_3.png'), ))

	probs = paddle.infer(
	output_layer=predict, parameters=parameters, input=test_data)
	lab = np.argsort(-probs)  # probs and lab are the results of one batch data
	print "Label of image/infer_3.png is: %d" % lab[0][0]	
	

if __name__ == "__main__":
	start = datetime.datetime.now()

	main()

	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'



