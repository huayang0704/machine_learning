#!/usr/bin/env python
# coding=utf8

import pandas as pd
import _0_config
import numpy as np
import sys


reload(sys)
sys.setdefaultencoding("utf-8")


#============================
# 对训练和测试数据进行转换清洗
#============================
data = pd.read_json(_0_config.originDataPath)
data = data['data']

jsonData = []
for index, row in data.iteritems():
    jsonData.append(row)

df = pd.DataFrame(jsonData)
tplData = df['tplData']


rows = []
for i, row in tplData.iteritems():
    if 'list' in row['sold_house']:
        rows = rows + row['sold_house']['list']

dfData = pd.DataFrame(rows)

df = dfData.reset_index()

# 增加整理一些字段
dfData['decoration'] = dfData['decorateType']
#====roomNum
dfData['bedroomNum'] = pd.to_numeric(dfData['roomNum'].replace(to_replace='[^\d].+', value='', regex=True))
# 处理异常数据
dfData = dfData[np.isfinite(dfData['bedroomNum'])]

# delete no use columns
willDeleteKeys = []
for column in dfData.columns:
    if column in _0_config.keys:
        continue
    else:
       willDeleteKeys.append(column)
dfData.drop(willDeleteKeys, 1, inplace=True)

#=====buildYear             2011
# filter build year is zero
dfData = dfData[dfData['buildYear'] != '0']
# transform string to int

#====decoration              其他(1)/毛坯(2)/简装(3)/精装(4)/...
temp = dfData.groupby(['decoration'])
keys = list(temp.groups.keys())
for i, val in enumerate(keys):
    dfData.loc[dfData['decoration'] == val,['decoration']] = i + 1


#====districtName            江北
# filter chongqing district
dfData = dfData[
        (dfData['districtName'] == '九龙坡') |
        (dfData['districtName'] == '南岸') |
        (dfData['districtName'] == '大渡口') |
        (dfData['districtName'] == '巴南') |
        (dfData['districtName'] == '江北') |
        (dfData['districtName'] == '江津') |
        (dfData['districtName'] == '沙坪坝') |
        (dfData['districtName'] == '渝中') |
        (dfData['districtName'] == '渝北')
]

#====九龙坡(1)/南岸(2)/...
temp = dfData.groupby(['districtName'])
keys = list(temp.groups.keys())
for i, val in enumerate(keys):
    dfData.loc[dfData['districtName'] == val,['districtName']] = i + 1


#====regionName          三峡广场(1)/上清寺(2)/...
temp = dfData.groupby(['regionName'])
keys = list(temp.groups.keys())
for i, val in enumerate(keys):
    dfData.loc[dfData['regionName'] == val,['regionName']] = i + 1

#====roomNum
# nothing


#====square  113
# nothing

#====price
# nothing

# write to file
df = dfData.reset_index()
df.to_json(path_or_buf=_0_config.cleanDataPath, orient='records')

#============================
# 对需要评估的房子进行转换清洗
#============================
data = pd.read_json(_0_config.salingDataPath)
df = pd.DataFrame(data)
df.drop(['url'], 1, inplace=True)
df = df.reset_index()
df.to_json(path_or_buf=_0_config.salingCleanDataPath, orient='records')
print('origin to clean successfully')
