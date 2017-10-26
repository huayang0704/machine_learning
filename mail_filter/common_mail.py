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
import numpy as np

sys.path.append('../')
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

    优化:
      1,概率取对数,防止四舍五入对概率的影响
      2,每个特征加1,防止单个特征概率为0
      3,概率累加

    :param train_matrix:[[1,0,1],[0,0,0]],要求每个样本的向量维度必须相同
    :param categories:一维列表,样本的分类结果,[1,0,...],分类结果只能为0,1,2,,,自然数
    :return:分类下各个特征的概率向量, 分类结果的概率向量,维数相同
    '''
    sum_docs = len(train_matrix)  # 总样本数
    num_words = len(train_matrix[0])  # 每个样本的特征数量

    # 计算每个分类的概率
    classify_l = sorted(list(set(categories)))
    classify_p = []
    for cla in classify_l:
        classify_p.append(
            len(filter(lambda x: True if x == cla else None, categories)) / float(sum_docs)
        )

    #每个特征在各分类下的概率
    single_vecs = []
    for cla in classify_l:
        single_vecs.append(np.ones(num_words))  # 初始化以单个样本长度的全为1的向量,目的是防止概率为0的情况

    p_sums = []
    for cla in classify_l:
        p_sums.append(num_words)  # 初始化总量,防止为0

    for i in range(len(categories)):
        single_vecs[categories[i]] += train_matrix[i]
        p_sums[categories[i]] += sum(train_matrix[i])

    # log()
    for i in range(len(single_vecs)):
       single_vecs[i] = np.log(single_vecs[i] / p_sums[i])  # 取对数,目的是很小指相乘四舍五入对结果影响很大

    # no-log()
    #for i in range(len(single_vecs)):
    #    single_vecs[i] = single_vecs[i] / p_sums[i]

    return single_vecs, classify_p


def classify_plus(classify_vec, p_vecs, ps):
    '''
    朴树贝叶斯预测分类,将各概率相加,P(a1|Y1) + P(a2|Y1) + ... + P(Y1)
    :param classify_vec: 待预测向量,跟p_vecs维数相同
    :param p_vecs: 各特征概率
    :param ps: 分类的概率
    :return:概率最大的分类
    '''
    result_p = []
    for i in range(len(ps)):
        result_p.append(sum(classify_vec * p_vecs[i]) + np.log(ps[i]))

    return result_p.index(np.max(result_p)), result_p

def classify_mult(classify_vec, p_vecs, ps):
    '''
    朴树贝叶斯预测分类,将个概率相乘,需要修改train_bayes方法,概率不取对数:P(a1|Y1) + P(a2|Y1) + ... + P(Y1)
    :param classify_vec: 待预测向量,跟p_vecs维数相同
    :param p_vecs: 各特征概率
    :param ps: 分类的概率
    :return:概率最大的分类
    '''
    result_p = []
    for i in range(len(ps)):
        if np.sum(classify_vec * p_vecs[i]) == 0:
            result_p.append(0.0)
        else:
            result_p.append(reduce(lambda x, y: x * y, filter(lambda x: x if x != 0 else None, classify_vec * p_vecs[i])) * ps[i])

    return result_p.index(np.max(result_p)), result_p

###--------------------------------------------------------------------------------------------

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
    分词,过滤掉英文字符数字标点符号单个词
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


def train_and_write(train_list, class_list, vec_path='para/p_vecs.txt', ps_path='para/ps.txt',
                    vocab_path='para/vocab_list.txt'):
    '''
    调用训练模型
    :param train_list: 样本数据,['我们的祖国越来越强大','祝贺十九大成功召开',...]
    :param class_list: 分类结果[0,1,...]
    :return:
    '''
    vocab_list = create_vocab_list(train_list)  # 所有词汇list
    print 'feature vector:', ",".join(vocab_list), len(vocab_list)
    train_matrix = []  # 向量矩阵
    for train in train_list:
        train_matrix.append(set_of_words_to_vec(vocab_list, train))

	print 'train_matrix:',train_matrix
    p_vecs, ps = train_bayes(np.array(train_matrix), class_list)
    print 'p_vesc:', p_vecs
    print 'ps:', ps
    # 保存参数
    file.save_parameters(vec_path, p_vecs)
    file.save_parameters(ps_path, ps)
    file.write_list(vocab_path, vocab_list)


def test_paramters(test_datas, vec_path='para/p_vecs.txt', ps_path='para/ps.txt"', vocab_path='para/vocab_list.txt'):
    '''
    测试,分类预测
    :return:
    '''
    p_vecs = file.load_parameters(vec_path)
    ps = file.load_parameters(ps_path)
    vocab_list = file.read_file_list(vocab_path)

    sum_nums = len(test_datas)
    error_num = 0.0
    for i in range(sum_nums):
        data = segment.segment(str(test_datas[i]))
        this_doc = np.array(set_of_words_to_vec(vocab_list, data))
        result, result_p = classify_plus(this_doc, p_vecs, ps)
        #result, result_p = classify_mult(this_doc, p_vecs, ps)

        if (int(test_datas[i].split("\t")[0]) != result):
            print result_p
            print (result),
            print test_datas[i]
            error_num += 1

    print 'error nums:', error_num, 'sum nums:', sum_nums
    print 'error rate:', round(error_num / sum_nums, 3)


if __name__ == "__main__":
    start = datetime.datetime.now()

    #data_list = file.read_file("test_train.data")
    data_list = file.read_file("train.data")
    print len(data_list)

    train_nums = 10000
    test_nums = 500

    # train data
    #train_data = data_list[0:train_nums]
    #train_data, class_data = load_data(train_data)
    #print train_data,class_data
    #train_list = word_segment(train_data)
    #for train in train_list:
    #    print ",".join(train)
    #train_and_write(train_list, class_data)

    # test
    test_paramters(data_list[train_nums:train_nums + test_nums])

    end = datetime.datetime.now()
    print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
