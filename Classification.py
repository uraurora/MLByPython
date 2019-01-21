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

class BPNN(Model):
    def __init__(self, ni, nh, no, active="tanh", iterations=5000, learnRate=0.001, B=0.002):
        # 调用父类赋值模型类型
        super(BPNN, self).__init__(modelType["BpClassification"], iterations)
        # self._type = modelType["BpClassification"]
        # 输入层，隐层和输出层的节点设置
        self.ni = ni
        self.nh = nh
        self.no = no
        # 设置各层的函数值存储向量
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        # 网络学习率、动量因子、迭代次数设定
        self.learnRate = learnRate
        self.B = B
        # self.iterations = iterations
        # 初始化权值矩阵
        self.wi = m.random.rand((self.ni, self.nh)).mat
        self.wo = m.random.rand((self.nh, self.no)).mat
        # 建立动量因子矩阵
        self.ci = m.fillMat((self.ni, self.nh)).mat
        self.co = m.fillMat((self.nh, self.no)).mat
        # 损失函数值记录
        self.error = []
        # 激活函数选择
        self.active = funDict[active]
        self.activeDiff = funDiffDict[active]

    def __str__(self):
        res = super(BPNN, self).__str__()
        res += "\t" + "Input dimension（输入维度）, Hidden dimension（隐层维度）, Output dimension（输出维度）:  "
        res += str((self.ni, self.nh, self.no)) + "\n"
        res += "\t" + "超参数（需要调节的参数）:  "
        res += "学习率：" + str(self.learnRate) + "\t" + "修正系数：" + str(self.B) + "\n"
        res += "\t" + "最终迭代的损失函数值:  "
        res += str(self.error[-1]) + "\n"
        return res

    def fit(self, inputs, outputs, bias=False, info=True, show=True):
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
        if info:
            self.__train()
        if show:
            # 画误随迭代次数差变化图
            try:
                from BaseCode.figure import ErrorPlot
                ErrorPlot(self.error, 1000, a=0.5, color='red')
            except:
                raise Exception("无法加载figure库，一般不会发生，注意看看")

    def __front(self, input):
        '''
        正向传播，对每行的input计算输出
        :return: 输出的实际值
        '''
        # 计算输入层输出
        if len(input) != self.ni:
            raise ValueError("输入节点数错误")
        for i in range(self.ni):
            self.ai[i] = input[i]
        # 计算隐层输出
        for j in range(self.nh):
            self.ah[j] = self.active(sum([self.ai[i]*self.wi[i][j] for i in range(self.ni)]))
        # 计算输入层输出
        for k in range(self.no):
            self.ao[k] = self.active(sum([self.ah[j]*self.wo[j][k] for j in range(self.nh)]))
        return self.ao

    def __back(self, output):
        '''
        反向传播，误差反向更新权重
        :return:
        '''
        Error = 0.0  # 网络误差
        if len(output) != self.no:
            raise ValueError('输出节点数错误')
        # 计算输出层误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            reo = self.ao[k]  # 实际输出
            eo = output[k]  # 期望输出
            output_deltas[k] = (eo - reo) * self.activeDiff(reo)
            Error += 0.5 * (eo - reo) ** 2  # 计算整体网络误差
        # 计算隐层误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            reo = self.ah[j]
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = self.activeDiff(reo) * error
        # 更新输出层权值
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += self.learnRate * change + self.B * self.co[j][k]
                self.co[j][k] = change
        # 更新隐层权重
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += self.learnRate * change + self.B * self.ci[i][j]
                self.ci[i][j] = change
        return Error

    def __train(self):
        for i in range(self.iterations):
            All_Error = 0.0
            for j in range(self.irow):
                self.__front(self._inputs[j])
                All_Error += self.__back(self._outputs[j])
            self.error.append(All_Error)
            if i % 100 == 0:
                print("iteration:" + str(i) + "\t" + "Lost-Function-error: " + str(All_Error))

    def saveWeights(self):
        try:
            import pandas as pd
            wi = pd.DataFrame(self.wi)
            wo = pd.DataFrame(self.wo)
            wi.to_csv("weights/bpwi.csv")
            wo.to_csv("weights/bpwo.csv")
        except:
            # raise Exception("无法加载pandas库，检查你有没有装吧，┓(;´_｀)┏")
            data = {"iterations": self.iterations,
                    "wi": [w for w in self.wi],
                    "wo": [b for b in self.wo],
                    "cost": str(self.error[-1])}
            f = open("weights/BpWeightsInfo.json", "w")
            json.dump(data, f)
            f.close()

    def loadWeights(self):
        f = open("weights/BpWeightsInfo.json", "r")
        data = json.load(f)
        f.close()
        cost = data["cost"]
        wi = [list(wi) for wi in data["wi"]]
        wo = [list(wo) for wo in data["wo"]]
        info = {"cost": cost, "wi": wi, "wo": wo}
        return info

    def fitWeights(self, wi, wo):
        '''填充权重矩阵，用来直接验证模型'''
        self.wi = wi
        self.wo = wo

    def predict(self, inputs, Debug=False):
        '''输入矩阵，进行预测'''
        if not isinstance(inputs, mat):
            raise TypeError("类型得是mat，_(:з」∠)_")
        data = copy.deepcopy(inputs)
        if Debug:
            try:
                import numpy as np
                ah = list(np.tanh(m.dot(data, mat(self.wi)).mat))
                ao = np.tanh(m.dot(mat(ah), mat(self.wo)).mat)
                return ao
            except:
                raise Exception("测试用numpy库加载失败，o(╥﹏╥)o")
        else:
            xwi = m.dot(data, mat(self.wi)).mat
            ah = mat([list(map(self.active, i)) for i in xwi])
            xwiwo = m.dot(ah, mat(self.wo)).mat
            ao = mat([list(map(self.active, i)) for i in xwiwo])
            return xwiwo


if __name__ == '__main__':

    # 先创建特征矩阵，默认规范化（数值在0-1之间）
    data_X = m.random.rand((100, 8))
    # data_X = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    # 创建特征的标签向量
    data_y = m.random.rand((100, 1))
    # data_y = [[0], [1], [0], [1], [1], [1], [1]]

    # 创建BP实例，输入矩阵列8， 隐含层矩阵列10， 输出层列1， 迭代次数5000， 梯度步长0.4， 修正参数0.2
    nn = BPNN(8, 6, 1, active="tanh", iterations=5000, learnRate=0.01, B=0.02)
    # 填充特征矩阵和标签， info表示是否显示迭代过程， show表示是否显示误差图像， bias偏置（为True，输入矩阵列需要加1）
    nn.fit(mat(data_X), mat(data_y), info=True, show=True, bias=False)
    # 存储权重矩阵，在weights文件夹下（需要pandas库）
    nn.saveWeights()
    # 显示算法基本信息
    print(nn)
    # 进行预测，如果Debug为True，则用到了numpy，否则则是自己写的
    print(nn.predict(m.random.rand((20, 8)), Debug=False))