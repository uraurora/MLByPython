#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 基础模型包，作为所有复杂机器学习模型的父类"""
__author__ = 'Sei Gao'
# date = 2019.1.16

from BaseCode.matrix import *
import BaseCode.matrix as m
import copy

class Model(object):

    def __init__(self, modelType, iterations):
        self.type = modelType
        self.iterations = iterations

    def __str__(self):
        res = "(*^▽^*)模型参数信息:  " + "\n"
        res += "\t" + "模型类型:  "
        res += str(self.type) + "\n"
        res += "\t" + "迭代次数:  "
        res += str(self.iterations) + "\n"
        return res

    def fit(self, inputs, outputs, bias):
        if not (isinstance(inputs, mat) and isinstance(outputs, mat)):
            raise TypeError("输入参数必须为mat类型")
        if bias:
            ones = m.fillMat((len(inputs.mat), 1), fill=1)
            i = copy.deepcopy(m.column_stack(inputs, ones))
        else:
            i = copy.deepcopy(inputs)
        o = copy.deepcopy(outputs)
        self._inputs = i.mat
        self._outputs= o.mat
        self.irow = i.row
        if self.irow != len(self._outputs):
            raise ValueError("输入行与标签行不一致")

    def saveWeights(self):
        pass

    def loadWeights(self):
        pass

    def _fitWeight(self):
        pass

    def predict(self):
        pass


modelType = {
    "base": "基础模型",
    "Linear": "线性模型",
    "BpRegression": "Bp神经网络回归",
    "BpClassification": "Bp神经网络分类",
    "NaiveBayes": "朴素贝叶斯分类",
    "DecisionTree": "决策树分类",
}
