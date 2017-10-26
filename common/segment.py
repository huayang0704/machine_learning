#!/usr/bin/python
# coding=utf8

##************************************************************************
## ** python 结巴中文分词器
## **
## **create: 2017-10-13
##************************************************************************
## ** qq: 876253250
## ** weibo: paul_华
##************************************************************************


import sys
import os
import datetime
import numpy as np
import jieba
import string
import re

# sys.path.append('../')
# from perceptron.perceptron import *
# from common.gradient import *

reload(sys)
sys.setdefaultencoding("utf-8")

# encoding
encoding = 'utf8'

# 停用词
stop_words = ['http']

# 中,英标点符号
filter_content = "[a-zA-Z0-9\s+\.\:\!\/_,$%^*(+\"\']+|[+——！，；。？、~@#￥%……&*（）]+"


def _filter_symbol(sentence):
    '''
    过滤中,英标点符号
    :param sentence:
    :return:
    '''
    return re.sub(filter_content.decode(encoding), "".decode(encoding), sentence)


def _filter_stop_words(seg, word_len=1):
    '''
    过滤停用词,默认过滤长度小于2的词
    :param seg: 分词结果,数组
    :param word_len:过滤长度<=word_len词组
    :return:过滤后的分词结果,数组
    '''
    if not seg or not word_len:
        raise Exception("seg is None")
    out = []
    for word in stop_words:
        for se in seg:
            if word not in se and len(se) > word_len:
                out.append(se)

    return out


def segment(sentence):
    '''
    中文分词
    :param sentence: 分词的内容
    :return:数组
    @jieba分词模式
    seg = jieba.cut(sentence, cut_all=True) #默认精确模式
    seg = jieba.cut_for_search(sentence) #搜索模式
    '''
    if not sentence:
        raise Exception("sentence is None")

    sentence = sentence.decode(encoding)
    sentence = _filter_symbol(sentence)
    seg = jieba.cut(sentence)  # 默认精确模式

    return _filter_stop_words(seg) if seg else []


if __name__ == "__main__":
    start = datetime.datetime.now()

    # s = "他来到了网易杭研大厦"
    # s = "你好吗？立夏了，心中又把你想起。愿会心的微笑，排满你每天的日历；抛弃压力，婴儿般惬意地呼吸；天气转热，别忘好好保重自己"
    s = '110,ok女子闯红灯被拦 脱鞋猛抽交警-图[腾讯]-中国移动冲浪助手:http://go.10086.cn/nd/lAz/cJctr'
    print ",".join(segment(s))


    end = datetime.datetime.now()
    print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
