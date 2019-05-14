#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 预处理包"""
__author__ = 'Sei Gao'
# date = 2019.1.27

from BaseCode.matrix import mat # 自写
import BaseCode.matrix as m
from BaseCode.Function import *
from BaseModel import *
import copy # 基本库
import json
try:
    import cPickle as pickle
except:
    import pickle

class Scaler(object):
    def __init__(self):
        pass

    @staticmethod
    def StandardScaler(dataX):
        '''标准化'''
        if isinstance(dataX, mat):
            x = copy.deepcopy(dataX)
        elif isinstance(dataX, list):
            x = copy.deepcopy(mat(dataX))
        else:
            raise TypeError("输入矩阵！！类型！！错误！！")
        pack = m.MeanAndVar(x)
        return mat([[(float(x.mat[i][j])-pack[j][0])/pack[j][1] for j in range(x.col)]
                    for i in range(x.row)])

    @staticmethod
    def MinMaxScaler(dataX):
        '''极差标准化'''
        if isinstance(dataX, mat):
            x = copy.deepcopy(dataX)
        elif isinstance(dataX, list):
            x = copy.deepcopy(mat(dataX))
        else:
            raise TypeError("输入矩阵！！类型！！错误！！")
        pack = m.MaxAndMin(x)
        return mat([[(float(x.mat[i][j])-pack[j][1])/(pack[j][0]-pack[j][1]) for j in range(x.col)]
                    for i in range(x.row)])

    @staticmethod
    def Normalizer(dataX):
        '''归一化'''
        if isinstance(dataX, mat):
            x = copy.deepcopy(dataX)
        elif isinstance(dataX, list):
            x = copy.deepcopy(mat(dataX))
        else:
            raise TypeError("输入矩阵！！类型！！错误！！")
        xT = x.T
        add = list(map(sum, [list(map(lambda x: x**2, i))for i in x.T.mat]))
        return mat([[(float(x.mat[i][j]**2)) / add[j] for j in range(x.col)]
                    for i in range(x.row)])


if __name__ == '__main__':
    a = [[3, 6, 9], [2, 3, 4], [1, 3, 6]]
    a = m.random.randint((1600, 100), (-200, 200))
    print Scaler.StandardScaler(a).shape
    print Scaler.Normalizer(a).shape
    print Scaler.MinMaxScaler(a).shape
	print "hello world"