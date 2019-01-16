# -*- coding:utf-8 -*-
# author = Sei
# date = 2018.09.08

# 优化包
from scipy.optimize import lsq_linear, least_squares
# 基础包
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

class Optimize:

    # region 线性与非线性模型最小二乘参数求解
    @staticmethod
    def linear_square(data_X, data_y, bounds=(-np.inf, np.inf)):
        '''
        功能：
            minimize 0.5 * ||A x - b||**2
                subject to lb <= x <= ub
            A为矩阵，b为向量（一维），x为求解对象，lb和ub限制x取值范围,存储在bounds元组中
        适合：
            用于求解线性回归的最优参数weight，即X*weight=y
        输入参数的要求：
            np.array格式
        返回值：
            打印具体信息的字典格式，算出的weight在['x']中,使用np.dot(test_X, w)可以算出测试集【test_x需要加一列1】
        '''
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        data_X = np.column_stack((data_X, np.ones(data_X.shape[0])))
        #test_X = np.c_[test_X, np.ones(test_X.shape[0])]

        res = lsq_linear(data_X, data_y, bounds, verbose=1)
        return res

    @staticmethod
    def least_square(min_function, data_X, data_y, x0):
        '''
        功能：
            解决非线性的一组函数的平方和的最小值，即weight = arg min(sum(min_func(data_x,y)**2,axis=0))

        适用：
            广义上的最小二乘法，即min((f(xi)-yi)**2)，即yi是真实值（训练集的标签），xi为训练集特征，f为模型值

        输入参数的要求：
            np.array格式

        :return:
            打印具体信息的字典格式，算出的weight在['x']中,使用np.dot(test_X, w)可以算出测试集【test_x需要加一列1】

        调用方式：
            res = least_squares(Optimize.fun, x0, args=(data_X, data_y))
        '''
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        data_X = np.column_stack((data_X, np.ones(data_X.shape[0])))

        return least_squares(min_function, x0, loss='soft_l1', args=(data_X, data_y))
    # endregion

class function:

    @staticmethod
    def error(w, *args):
        # 误差函数，包含自己定义的模型函数
        data_x, y = args
        # 定义模型函数
        model_value = np.dot(data_x, w)
        return function.tanh(model_value - y)

    @staticmethod
    def MSE(w, X, y):
        weight = np.array(w)
        return np.dot(X, weight) - y

    # region 常用非线性函数
    @staticmethod
    def sigmod(x):
        # 非线性激活函数sigmod
        return float(1) / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        # 非线性激活函数双曲正切tanh
        return np.tanh(x)

    @staticmethod
    def step(x):
        # 阶跃函数
        x = np.array(x)
        return np.array(list(map(lambda var: 1 if var > 0.35 else 0, x)))
    # endregion

if __name__ == '__main__':
    np.random.seed(1)
    data_X = np.array([[1.24, 1.27], [1.36, 1.74], [1.38, 1.64], [1.38, 1.82], [1.38, 1.90], [1.40, 1.70], [1.48, 1.82],
                       [1.54, 1.82], [1.56, 2.08], [1.14, 1.82], [1.18, 1.96], [1.20, 1.86], [1.26, 2.00], [1.28, 2.00],
                       [1.30, 1.96]])

    x0 = np.array([0, 0, 0])
    data_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    #data_y = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
    test_X = [[1.24, 1.80],[1.28, 1.84],[1.40, 2.04],[1.48, 1.56],[1.59, 1.93],[1.37, 1.83]]
    test_X = np.c_[test_X, np.ones(np.array(test_X).shape[0])]
    res = Optimize.least_square(function.error, data_X, data_y, x0)
    w = res['x']
    data_X = np.column_stack((data_X, np.ones(data_X.shape[0])))
    test_y = np.dot(test_X, w)

    print('验证:', np.dot(data_X, w))
    print('测试:', test_y)





