#!/usr/bin/env python
# coding=utf8


import pandas as pd
import _0_config
import sys


reload(sys)
sys.setdefaultencoding("utf-8")

# 对数据进行分配
# 首先要对数据进行打乱
# 使用2/3的数据进行训练
# 使用1/3的数据进行测试

df = pd.read_json(path_or_buf=_0_config.cleanDataPath)


# random sort row
df = df.sample(frac=1)
df = df.reset_index(level=None, drop=True).drop(['index'], 1, inplace=False)


totalRowCount = len(df.index)
print('total row count:' + str(totalRowCount))
trainCount = int(totalRowCount * _0_config.ratio)
print('train count:' + str(trainCount))

testCount = totalRowCount - trainCount
print('test count:' + str(testCount))

trainData = df[0:trainCount]
trainData.to_json(path_or_buf= _0_config.trainDataPath)

testData = df[trainCount:totalRowCount]
testData.to_json(path_or_buf = _0_config.testDataPath)


#============================
# 对需要评估的房子进行转换清洗
#============================
data = pd.read_json(_0_config.salingCleanDataPath)
df = pd.DataFrame(data)
transedData = []
for index, row in df.iterrows():
    if _0_config.decorations.has_key(row['decoration']):
        row['decoration'] = _0_config.decorations[row['decoration']]
        row['districtName'] = _0_config.districtName[row['districtName']]
        row['regionName'] = _0_config.regionName[row['regionName']]
        transedData.append(row)

df = pd.DataFrame(transedData)
df = df.reset_index()
df.to_json(path_or_buf=_0_config.predictDataPath)
print('train and test data ready!')
