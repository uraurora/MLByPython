#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 常用函数包"""
__author__ = 'Sei Gao'
# date = 2019.01.14

import matrix as m
from matrix import mat
import math
import random

def sigmoid(x):
    return float(1.0)/(1 + math.exp(-x))
    # return 0.5 * (1 + math.tanh(0.5 * x))

def tanh(x):
    return math.tanh(x)

def relu(x):
    if x < 0:
        return 0
    else:
        return x

funDict = {
    "sigmod": sigmoid,
    "tanh": tanh,
}

def sigmoidDiff(x):
    '''
    输入的是算出来的函数值
    :param x:
    :return:
    '''
    return x * (1 - x)
    # return 0.5 * (1 + math.tanh(0.5 * x))

def tanhDiff(x):
    '''
    输入的是算出来的函数值
    :param x:
    :return:
    '''
    return 1 - x ** 2


funDiffDict = {
    "sigmod": sigmoidDiff,
    "tanh": tanhDiff,
}


