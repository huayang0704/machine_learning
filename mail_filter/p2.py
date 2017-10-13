#!/usr/bin/python
#coding=utf8

##************************************************************************
## ** 模板脚本
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

#sys.path.append('../')
from perceptron.perceptron import *
from common.gradient import *

reload(sys)
sys.setdefaultencoding("utf-8")

# -*- coding: cp936 -*-
"""
基于支持向量机的邮件分类系统
使用0,1标记词的出现与否
author :luchi
data :1025/11/29 
"""
import re
from math import *
from SVMKernel import *
#切割文本,统计词频
def splitText(bigString):
   
	wordlist={}
	rtnList=[]
	wordFreqList={}
	#分词
	listofTokens=re.split(r'\W*',bigString)
	length=len(listofTokens)
	for token in listofTokens:
		if  not wordlist.has_key(token):
			wordlist[token]=1
			rtnList.append(token)
		else:
			wordlist[token]+=1
		wordFreqList[token]=float(wordlist[token])/length
	return rtnList,wordFreqList
	   
#统计单词反文档频率
def docFre(word):
	fre=0
	for i in range(1,26):
		if word in re.split(r'\W*',open('spam/%d.txt' % i).read()):
			 fre+=1
   
	return float(fre)/25

	

#特征词提取，这里面使用TF-IDF方法
	

def extractFeature(textType):
	docList=[];classList=[];fullText=[]
	wordTFIDF={}
	#每个类测试邮件一共有25封
	for i in range(1,26):
		wordlist,wordFreqList=splitText(open(textType+'/%d.txt' % i).read())
		fullText.append(wordlist)
		for word in wordFreqList:
			wordIDF=docFre(word)
			wordTFIDFValue=wordIDF*wordFreqList[word]
			if not  wordTFIDF.has_key(word):
				wordTFIDF[word]=wordTFIDFValue
			else :
				wordTFIDF[word]+=wordTFIDFValue
	sortedWordTFIDF=sorted(wordTFIDF.iteritems(),key=lambda asd:asd[1],reverse=True)
   #选取前100个词为分类词
	keywords=[word[0] for word in sortedWordTFIDF[:100]]
	return keywords

#对一个邮件词集构建特征向量（使用0,1表示存在与否）
def extaxtDocFeatureVec(text,keyword):
	vec=[]
	for i,word in enumerate(keyword):
		if word in text:
			vec.append(1)
		else :
			vec.append(0)
	return vec

#抽取所有邮件的特征向量
def extactFeatureVec():
	hamWordsVec=extractFeature('ham')
	spamWordsVec=extractFeature('spam')
	wordVecs=[]
	classList=[]
	for i in range(1,26):
		wordlistHam,wordFreqList=splitText(open('ham/%d.txt' % i).read())
		wordlistSpam,wordFreqList=splitText(open('spam/%d.txt' % i).read())
		vecHam=extaxtDocFeatureVec(wordlistHam,hamWordsVec)
		vecSpam=extaxtDocFeatureVec(wordlistSpam,spamWordsVec)
		wordVecs.append(vecHam)
		classList.append(1)
		wordVecs.append(vecSpam)
		classList.append(-1)
##	print wordVecs
##	print classList
		
	return wordVecs,classList

#使用SVM训练数据并使用交叉测试测试正确率
def textSpam(k1=1.3):
	dataArr,labelArr=extactFeatureVec()
	trainDataArr=dataArr[:40]
	trainLabelArr=labelArr[:40]
	b,alphas = smoP(trainDataArr, trainLabelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
	datMat=mat(trainDataArr); labelMat = mat(trainLabelArr).transpose()
	svInd=nonzero(alphas.A>0)[0]
	sVs=datMat[svInd] #get matrix of only support vectors
	labelSV = labelMat[svInd];
	print "there are %d Support Vectors" % shape(sVs)[0]
	testDataMat=mat(dataArr[39:-1])
	testLabel=labelArr[39:-1]
##	testLabel[2]=testLabel[2]*-1
	m,n = shape(testDataMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs,testDataMat[i,:],('rbf', k1))
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		if sign(predict)!=sign(testLabel[i]): errorCount += 1
	print "the training error rate is: %f" % (float(errorCount)/m)
  

if __name__ == "__main__":
	start = datetime.datetime.now()

	textSpam()	 

	end = datetime.datetime.now()
	print '[' + end.now().strftime('%Y-%m-%d %H:%M:%S') + ']******模型运行时长为:' + str((end - start).seconds) + '秒 *********'
