#!/usr/bin/env python
# coding=utf8

# Arthor: YouShaohua
# Date  : Sep. 15 2017


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager, FontProperties
import _0_config

def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

#=====================================
# 此文件用于画图，比较挂牌价格和评估价格
#=====================================

# 读取挂牌价格
evaling = pd.read_json(path_or_buf=_0_config.predictDataPath)
if evaling.has_key('price'):
	evaling_price = evaling['price']

# 读取模型评估价格
evaled= pd.read_json(path_or_buf=_0_config.resultDataPath)
evaled_price = evaled.price

# 画图
rowLen = len(evaled_price)
x_list = np.arange(rowLen) + 1

font = {
    'family': 'serif',
    'color':  'dark',
    'weight': 'normal',
    'size': 16,
}
plt.title('重庆市二手房评估（30间）- 评估价与挂牌价', fontproperties=getChineseFont())
plt.xlabel('房子', fontproperties=getChineseFont())
plt.ylabel('房价', fontproperties=getChineseFont())
plt.plot(x_list,evaling_price)
plt.plot(x_list, evaled_price)
plt.show()




