#!/usr/bin/python
# coding=utf8

##************************************************************************
## ** 垃圾邮件过滤
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

reload(sys)
sys.setdefaultencoding("utf-8")



# 数据集
def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please',"dog"],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not

    return posting_list, class_vec


# 创建词汇表
def create_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)  # union of two sets

    return list(vocab_set)


# 词向量
def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word %s is not in my Vocabulary" % word

    return return_vec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # print 'p1:', p1, 'p0:', p0
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    #load data,样本数据,分类结果
    listOPosts, listClasses = load_data_set()
    #print 'listOPosts:', listOPosts  # 训练样本数据,[[]]
    #print 'listClasses:', listClasses  # 样本结果[]

    myVocabList = create_vocab_list(listOPosts)  # 所有词汇list
    # print 'myVocabList:', myVocabList
    trainMat = []  # 向量矩阵
    for postinDoc in listOPosts:
        # trainMat.append(set_of_words_to_vec(myVocabList, postinDoc))
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print 'trainMat:', array(trainMat)
    print 'p0V:', p0V
    print 'p1V:', p1V
    print 'pAb:', pAb
    testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(set_of_words_to_vec(myVocabList, testEntry))
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    # thisDoc = array(set_of_words_to_vec(myVocabList, testEntry))
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

def read_file(file_path):
    '''
    按行读取文本
    :param file_path:
    :return:
    '''
    file = open(file_path)
    datas = list()
    while True:
        lines = file.readlines(100000)
        datas.extend(lines)
        if not lines:
            break

    file.close()

    return datas

def load_data(datas):
    train_list = []
    class_list = []
    for data in datas:
        class_list.append(int(data.split("\t")[0]))
        train_list.append(data.split("\t")[1])

    return train_list,class_list

def word_segment(train_datas):
    train_list = []
    for train in train_datas:
        train_list.append(segment.segment(train))

    return train_list

def train_and_test(listOPosts, listClasses):
    '''

    :param listOPosts: 样本数据
    :param listClasses: 分类结果
    :param testEntry: 测试数据
    :return:
    '''

    myVocabList = create_vocab_list(listOPosts)  # 所有词汇list
    # print 'myVocabList:', myVocabList
    trainMat = []  # 向量矩阵
    for postinDoc in listOPosts:
        # trainMat.append(set_of_words_to_vec(myVocabList, postinDoc))
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))

    # print 'trainMat:', trainMat


    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # print 'trainMat:', array(trainMat)
    print 'p0V:', p0V
    print 'p1V:', p1V
    print 'pAb:', pAb
    ## test
    test_datas = datas[50:53]
    for i in range(len(test_datas)):
        data = segment.segment(str(test_datas[i]))
        thisDoc = array(bagOfWords2VecMN(myVocabList, data))
        print (classifyNB(thisDoc, p0V, p1V, pAb)),
        print test_datas[i]


if __name__ == "__main__":
    start = datetime.datetime.now()

    #load data

    # posting_list,class_vec = load_data_set()
    # vocab_set = create_vocab_list(posting_list)
    # print vocab_set
    # input_set = set(['my','ok','food'])
    # return_vec = set_of_words_to_vec(vocab_set,input_set)
    # print return_vec

    #testingNB()

    datas = read_file("train.txt")
    train_datas = datas[0:50]

    train_datas,class_datas = load_data(train_datas)
    train_list = word_segment(train_datas)
    #
    # test = str(datas[51:52])
    # print 'test:', datas[51:52][0]
    # test_datas = segment.segment(test)

    # print train_list
    # print class_datas
    # print segment.segment(train_datas[0])
    train_and_test(train_list, class_datas)


    end = datetime.datetime.now()
    print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
