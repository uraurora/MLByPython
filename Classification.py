#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 我的分类操作包，包括一些分类方法"""
__author__ = 'Sei Gao'
# date = 2019.1.16

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

class NaiveBayes(Model):
    def __init__(self, iterations):
        # 调用父类构造函数，得到模型类型和迭代次数
        super(NaiveBayes, self).__init__(modelType["NaiveBayes"], iterations)