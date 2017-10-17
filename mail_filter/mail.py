#!/usr/bin/python
# coding=utf8

##************************************************************************
## ** 垃圾短信过滤
## **
## **
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************


import sys
import os
import datetime
from numpy import *

sys.path.append('../')
# from perceptron.perceptron import *
# from common.gradient import *
from common import segment
from common import file_tool as file

reload(sys)
sys.setdefaultencoding("utf-8")


def train_bayes(train_matrix, categories):
    '''
    朴素贝叶斯训练过程:
      1,创建去重的特征列表,['','','',...]
      2,按照特征列表将样本向量化,[[0,1,1],[2,1,0]...],对应的分类结果[0,1,...]
      3,统计每个特征在各个分类情况下的概率,p(特征A|y=0) = 特征A在y=0分类下出现的次数 / 分类下0下所有特征的数量
      4,预测分类:样本的每个特征的概率相乘 * 分类的概率,取最大概率即为分类结果

    :param train_matrix:[[1,0,1],[0,0,0]],要求每个样本的向量维度必须相同
    :param categories:一维列表,样本的分类结果,[1,0,...]
    :return:分类为0,1下各个特征的概率向量,分类为1的概率,p1_vec,p0_vec与样本的维度相同
    '''
    sum_docs = len(train_matrix)  # 总样本数
    num_words = len(train_matrix[0])  # 每个样本的特征数量
    p1 = sum(categories) / float(sum_docs)  # 分类为1的概率
    single_p0_vec = ones(num_words)  # 初始化以单个样本长度的全为1的向量,目的是防止概率为0的情况
    single_p1_vec = ones(num_words)
    p0_sum = 2.0  # 初始化总量,防止为0
    p1_sum = 2.0
    for i in range(sum_docs):
        if categories[i] == 1:
            single_p1_vec += train_matrix[i]
            p1_sum += sum(train_matrix[i])
        else:
            single_p0_vec += train_matrix[i]
            p0_sum += sum(train_matrix[i])

    #print 'single_p1_vec:', single_p1_vec
    #print 'single_p1_vec / p1_sum:', single_p1_vec / p1_sum
    #print 'single_p0_vec:', single_p0_vec
    #print 'single_p0_vec / p0_sum', single_p0_vec / p0_sum
    p1_vec = log(single_p1_vec / p1_sum)  # 取对数
    p0_vec = log(single_p0_vec / p0_sum)
    #print 'p1_vec', p1_vec
    #print 'p0_vec:', p0_vec
    return p0_vec, p1_vec, p1


def classify(classify_vec, p0_vec, p1_vec, p1):
    '''
    朴树贝叶斯预测分类
    :param classify_vec: 待预测向量,跟p0_vec,p1_vec维数相同
    :param p0_vec: 分类为0的各特征概率
    :param p1_vec: 分类为1的各特征概率
    :param p1: 分类为1的概率
    :return:0 or 1
    '''
    p1 = sum(classify_vec * p1_vec) + log(p1)
    p0 = sum(classify_vec * p0_vec) + log(1.0 - p1)
    if p1 > p0:
        return 1
    else:
        return 0


def create_vocab_list(data_set):
    '''
    创建全分词列表(去重)
    :param data_set: 输入数据格式[['祖国','强大','幸福'],['幸福']]
    :return:去重词组列表['祖国','强大','幸福']
    '''
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)

    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    '''
    词袋模型,对输入数据,创建词向量
    :param vocab_list: 全词向量,['祖国','强大','幸福']
    :param input_set: 输入列表,['幸福', '富强']
    :return:词向量,[0,0,1]
    '''
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1

    return return_vec


def word_segment(train_datas):
    '''
    分词
    :param train_datas:
    :return:
    '''
    train_list = []
    for train in train_datas:
        train_list.append(segment.segment(train))

    return train_list


def load_data(datas):
    '''
    获取训练数据和样本标签
    :param datas:
    :return:
    '''
    trains = []
    classes = []
    for data in datas:
        classes.append(int(data.split("\t")[0]))
        trains.append(data.split("\t")[1])

    return trains, classes


def train_and_write(train_list, class_list):
    '''
    调用训练模型
    :param listOPosts: 样本数据
    :param listClasses: 分类结果
    :param testEntry: 测试数据
    :return:
    '''
    vocab_list = create_vocab_list(train_list)  # 所有词汇list
    #print 'feature vector:', vocab_list, len(vocab_list)
    train_matrix = []  # 向量矩阵
    for train in train_list:
        train_matrix.append(set_of_words_to_vec(vocab_list, train))

    print 'train data matrix:', train_matrix
    p0V, p1V, p1 = train_bayes(array(train_matrix), array(class_list))
    # print 'p0V:', p0V
    # print 'p1V:', p1V
    # print 'pAb:', pAb
    # 保存参数
    file.save_parameters("p0.txt", p0V)
    file.save_parameters("p1V.txt", p1V)
    file.save_parameters("pAb.txt", array([p1]))
    file.write_list("vocabList.txt", vocab_list)


def test_paramters(test_datas):
    '''
    测试,分类预测
    :return:
    '''
    p0V = file.load_parameters("p0.txt")
    p1V = file.load_parameters("p1V.txt")
    p1 = file.load_parameters("pAb.txt")
    vocab_list = file.read_file_list("vocabList.txt")

    sum_nums = len(test_datas)
    error_num = 0.0
    for i in range(sum_nums):
        data = segment.segment(str(test_datas[i]))
        this_doc = array(set_of_words_to_vec(vocab_list, data))
        result = classify(this_doc, p0V, p1V, p1)

        if (int(test_datas[i].split("\t")[0]) != result):
            print (result),
            print test_datas[i]
            error_num += 1

    print 'error nums:', error_num, 'sum nums:', sum_nums
    print 'error rate:', round(error_num / sum_nums, 3)


if __name__ == "__main__":
    start = datetime.datetime.now()

    data_list = file.read_file("train.txt")
    print len(data_list)

    train_nums = 2
    test_nums = 1

    # train data
    train_data = data_list[0:train_nums]
    train_data, class_data = load_data(train_data)
    train_list = word_segment(train_data)
    train_and_write(train_list, class_data)

    #test
    test_paramters(data_list[train_nums:train_nums + test_nums])

    end = datetime.datetime.now()
    print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
