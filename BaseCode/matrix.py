#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    矩阵运算的函数集合，包括
    （1）实例初始化：
        mat(A),会得到矩阵的行和列，内部保存传进来的mat，对其进行操作，会自动判断A的数据类型
    （2）运算符：
        矩阵加减法：+、-（支持连加连减）
        矩阵对应元素乘法：*（支持连乘）
        打印：形式为'shape:(row,col),data:矩阵列表'
    （3）属性方法：
        矩阵转置：调用方式：mat(A).T
        非奇异矩阵的逆：:mat(A).I
        非奇异矩阵的行列式值：mat(A).det
        非奇异矩阵的三角分解：LU(matrix, isPart=False)，isPart为真时返回一个L，U上下三角矩阵组成的元组，为False返回LU组成的一个矩阵
        矩阵的迹：mat.tr

    （3）静态方法：
        生成单位矩阵：ones
        矩阵乘常数：dotk(k, mat)，注意第一个参数要是常数
        矩阵向量1范数:norm_1
        矩阵向量无穷范数:norm_inf
        # TODO
        初等行变换：（未完成）

    （4）类成员函数（protected）：
        第一个非零数归一化:
        向量和：_vecAdd
        向量减：_vecSub


    # :TODO矩阵乘法：dot（支持多参数连乘，但要注意数据类型）
    矩阵交换行：swap

    gzha:
    date: 2018.04.07
    矩阵换行：exchange_hang
    矩阵绝对值最大的元素：absmax_mat
    选主元特用的：absmaxsp_mat
    高斯消去法求逆：mat_gaosi

    矩阵的定义：此处矩阵的定义
                矩阵都是由一个二维列表表示的即[[]]的形式，具体例子如[[1,12],[2,1],[0,2],[1,3]],
                此为4*2的矩阵，4为行，2为列，推荐用len(mat)表示的是行数，len(mat[0])用来表示列数
'''
__author__ = 'Sei Gao'
# date = 2018.03.30

import copy
import math
import time
import random as rd
import numpy as np
#sys.setrecursionlimit(150)  # set the maximum depth as 1500


class mat:
    # region 运算符逻辑
    def __init__(self, matrix):
        if isinstance(matrix, list):
            self.row = len(matrix)
            self.col = len(matrix[0])
            self.shape = (self.row, self.col)
            self.mat = matrix

        elif isinstance(matrix, mat):
            self.row = len(matrix.mat)
            self.col = len(matrix.mat[0])
            self.shape = (self.row, self.col)
            self.mat = matrix.mat
        else:
            raise TypeError("矩阵基础类型必须为列表或mat类")

    def __str__(self):
        return "shape:" + str(self.shape) + ",data:" + str(self.mat)

    # TODO:连加矩阵未追加判断
    def __add__(self, other, *mat_else):
        if not isinstance(other, mat):
            raise TypeError("数据类型不一致")
        # 判断矩阵规格是否一致
        if self.shape != other.shape:
            assert "用你脑子想想，矩阵维度不一样能加么？"
        ret_mat = [None] * self.row
        for i in range(len(ret_mat)):
            ret_mat[i] = [0] * self.col
        for i in range(self.row):
            for j in range(self.col):
                ret_mat[i][j] = self.mat[i][j] + other.mat[i][j]
                for k in range(len(mat_else)):
                    ret_mat[i][j] += mat_else[k][i][j]
        return mat(ret_mat)

    def __sub__(self, other, *mat_else):
        if not isinstance(other, mat):
            raise TypeError("数据类型不一致")
        # 判断矩阵规格是否一致
        if self.shape != other.shape:
            assert "用你脑子想想，矩阵维度不一样能减么？"
        ret_mat = [None] * self.row
        for i in range(len(ret_mat)):
            ret_mat[i] = [0] * self.col
        for i in range(self.row):
            for j in range(self.col):
                ret_mat[i][j] = self.mat[i][j] - other.mat[i][j]
                for k in range(len(mat_else)):
                    ret_mat[i][j] -= mat_else[k][i][j]
        return mat(ret_mat)

    # 注意啦，这个是对应元素相乘，不是矩阵点乘！
    def __mul__(self, other, *mat_else):
        ret_mat = [None] * self.row
        for i in range(len(ret_mat)):
            ret_mat[i] = [0] * self.col
        if isinstance(other, mat):
            # 判断矩阵规格是否一致
            if self.shape != other.shape:
                assert "用你脑子想想，矩阵维度不一样能乘么？"
            for i in range(self.row):
                for j in range(self.col):
                    ret_mat[i][j] = self.mat[i][j] * other.mat[i][j]
                    for k in range(len(mat_else)):
                        ret_mat[i][j] *= mat_else[k][i][j]
            return mat(ret_mat)

    # endregion

    # region 属性方法
    # 转置操作(这样调用：mat(a).T)
    @property
    def T(self):
        # # 初始化矩阵
        # ret_mat = [None] * self.col
        # for i in range(len(ret_mat)):
        #     ret_mat[i] = [0] * self.row
        # # 元素转置
        # for i in range(self.col):
        #     for j in range(self.row):
        #         ret_mat[i][j] = self.mat[j][i]
        # return mat(ret_mat)
        return mat(map(list, zip(*self.mat)))

    # 求行列式操作(这样调用：mat(a).det)
    @property
    def det(self):
        '''
         矩阵求行列式的值:建议采用以下的形式调用此函数    mat.det
         return:
             行列式的值
        '''
        det = 1
        LU = copy.deepcopy(mat.LU(self.mat, False))
        for i in range(self.row):
            det *= LU.mat[i][i]
        return det

    # region 求逆操作
    @property
    def I(self):
        '''
        利用三角分解求解矩阵的逆
        :return: 矩阵的逆
        '''
        matrix = copy.deepcopy(self.mat)
        L, U = mat.LU(matrix, True)
        n = len(L)
        # LU逆矩阵初始化
        L_I = [None] * n
        U_I = [None] * n
        for i in range(n):
            L_I[i] = [0] * n
            U_I[i] = [0] * n
        for i in range(n):
            L_I[i][i] = 1
            if U[i][i] == 0:
                raise Exception("奇异矩阵警告，吔我大便玉啦！")
            U_I[i][i] = 1.0 / U[i][i]
        # L矩阵求逆
        for j in range(n - 1):
            for i in range(j + 1, n):
                qiuhe = 0
                for k in range(j + 1, i):
                    qiuhe = qiuhe + L[i][k] * L_I[k][j]
                L_I[i][j] = -L[i][j] - qiuhe
        # U矩阵求逆
        for j in range(n - 1, 0, -1):
            for i in range(j - 1, -1, -1):
                qiuhe = 0
                for k in range(j, i, -1):
                    qiuhe = qiuhe + U[i][k] * U_I[k][j]
                U_I[i][j] = -float(qiuhe) / U[i][i]
        return dot(U_I, L_I)
    # endregion

    # TODO:并不是赫米特标准型
    @property
    def hermite(self, isdet=False):
        '''
        求矩阵的Hermite标准型
        :return: 返回mat类的倒三角矩阵
        :param isdet: bool型数据，若为true则不进行对角归一,可以用于求行列式值
        '''
        ret_mat = self.mat
        if isdet:
            ret_mat[0] = self.mat[0]
            for i in range(1, self.row):
                main_row = ret_mat[i - 1]
                for j in range(i, self.row):
                    temp_row = ret_mat[j]
                    k = float(main_row[i - 1]) / temp_row[i - 1]
                    temp_row = self._vecSub(temp_row, self._vecMul_k(main_row, k))
                    ret_mat[j] = temp_row
        else:
            ret_mat[0] = self._vecPro(self.mat[0])
            for i in range(1, self.row):
                main_row = ret_mat[i - 1]
                for j in range(i, self.row):
                    temp_row = ret_mat[j]

                    k = temp_row[i - 1]
                    m = self._vecMul_k(main_row, k)
                    temp_row = self._vecSub(temp_row, m)
                    # temp_row = self._vecSub(temp_row, self._vecMul_k(main_row, temp_row[i - 1]))
                    ret_mat[j] = self._vecPro(temp_row)
                    # 如果此行变为0，则交换此行
                    # if temp_row == [0]*self.row:
                    #     for item in range(self.row, 0, -1):
                    #         if ret_mat[item] != [0]*self.row:
                    #             self.swap(ret_mat, j, item)
        return mat(ret_mat)

    @property
    def tr(self):
        tr = 0
        for i in range(self.row):
            tr += self.mat[i][i]
        return tr

    @property
    def maxFeature(self, vk, k=pow(10, 4)):
        '''
        根据数值幂迭代法求解最大的特征值
        :param vec: 初始向量
        :param k: 迭代终止条件
        :return: 返回最大的正特征值与特征向量的元组
        '''
        vk = mat(vk)
        u = dot(self.mat, vk)
        m = max(u.mat)[0]
        vm = mat.dotk(1.0 / m, u)
        i = 0
        while i < k:
            u = dot(self.mat, vm)
            m = max(u.mat)[0]
            vk = mat.dotk(1.0/m, u)
            i += 1
        return m, vk.mat



    # endregion

    # region 类私有方法
    @classmethod
    def _vecAdd(cls, vec1, vec2):
        '''
        向量和
        :param vec1:
        :param vec2:
        :return:
        '''
        return list(map(lambda x, y: x+y, vec1, vec2))
    @classmethod
    def _vecSub(cls, vec1, vec2):
        '''
        向量减
        :param vec1:
        :param vec2:
        :return:
        '''
        return list(map(lambda x, y: x - y, vec1, vec2))
    @classmethod
    def _vecMul_k(cls, vec, k):
        '''
        向量乘常数
        :param vec:
        :param k:
        :return:
        '''
        return list(map(lambda x: x * k, vec))
    @classmethod
    def _vecPro(cls, vec):
        '''
        第一个非零数归一化
        :param vec:
        :return:
        '''
        for i in vec:
            if i != 0:
                k = i
                return list(map(lambda x: float(x)/k, vec))
        return vec

    # endregion

    #region 静态方法
    @staticmethod
    def ones(n):
        '''
        返回一个n维的单位矩阵
        :param n:如果是整数，则是n维的单位矩阵
        :return:
        '''
        if isinstance(n, int):
            I = [None] * n
            for i in range(len(I)):
                I[i] = [0] * n
            for i in range(n):
                I[i][i] = 1.0
            return mat(I)
        else:
            raise TypeError, "必须是整数"

    @staticmethod
    def norm_1(matrix):
        '''
         矩阵求一范数
             mat:矩阵， （任意行列的矩阵均可计算，行列向量亦可）
             return:
                 矩阵一范数
        '''
        m = mat(matrix)
        ret = []
        for i in range(m.col):
            num = 0
            for j in range(m.row):
                num += abs(m.mat[j][i])
            ret.append(num)
        return max(ret)

    @staticmethod
    def norm_inf(matrix):
        '''
         矩阵求无穷范数
             mat:矩阵， （任意行列的矩阵均可计算，行列向量亦可）
             return:
                 矩阵无穷范数
        '''
        m = mat(matrix)
        ret = []
        for i in range(m.row):
            num = 0
            for j in range(m.col):
                num += abs(m.mat[i][j])
            ret.append(num)
        return max(ret)

    @staticmethod
    def LU(matrix, isPart=False):
        '''
         矩阵LU分解
             mat:矩阵 （非奇异矩阵）
             return:
                 LU上下三角的矩阵的元组
        '''
        matrix = mat(matrix)
        if matrix.row != matrix.col:
            raise Exception("奇异矩阵警告！")
        n = matrix.row
        m = copy.deepcopy(matrix.mat)
        for k in range(n):
            for j in range(k, n):
                temp = 0
                for t in range(k):
                    temp += m[k][t] * m[t][j]
                m[k][j] = matrix.mat[k][j] - temp
            for i in range(k+1, n):
                temp = 0
                for t in range(k):
                    temp += m[i][t] * m[t][k]
                if m[k][k] == 0:
                    raise Exception("奇异矩阵，无法三角分解")
                m[i][k] = float(matrix.mat[i][k] - temp)/m[k][k]
        if isPart == True:
            L = [None] * n
            for i in range(len(L)):
                L[i] = [0] * n
            U = [None] * n
            for i in range(len(U)):
                U[i] = [0] * n
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        L[j][i] = 1
                    else:
                        L[j][i] = m[j][i]
            for i in range(n):
                for j in range(i, n):
                    U[i][j] = m[i][j]
            return L, U
        return mat(m)
    #endregion

class random:
    @staticmethod
    def rand(shape):
        '''
        返回shape形状的随机数的mat矩阵，其中元素全都属于0-1之间
        :param shape: 元组
        :return: mat矩阵
        '''
        if not isinstance(shape, tuple):
            raise TypeError, "必须是元组类型，（行，列）"
        row, col = shape
        return mat([[rd.random() for i in range(col)]for j in range(row)])

    @staticmethod
    def randint(shape, rangesize):
        '''
        返回shape形状的随机整数mat矩阵，整数范围由元组rangesize来定
        :param shape:
        :param rangesize:
        :return:
        '''
        if not isinstance(shape, tuple) and isinstance(range, tuple):
            raise TypeError, "参数必须是元组类型，（行，列）"
        row, col = shape
        m, n = rangesize
        return mat([[rd.randint(m, n) for i in range(col)]for j in range(row)])


# 矩阵乘法，支持连乘操作（这样调用：matrix.dot(A, B)）
def dot(matrix, other, *mat_else):
    matA = mat(matrix)
    matB = mat(other)
    # 判断相乘矩阵的规格
    if matA.col == matB.row:
        temp = 0
        # 初始化返回矩阵
        ret_mat = [None] * matA.row
        for i in range(len(ret_mat)):
            ret_mat[i] = [0] * matB.col
        # 对每个元素赋值

        for i in range(matA.row):
            for m in range(matB.col):
                for j in range(matA.col):
                    temp += matA.mat[i][j] * matB.mat[j][m]
                ret_mat[i][m] = temp
                temp = 0
        temp_mat = mat(ret_mat)
        for item in mat_else:
            temp_mat = dot(temp_mat, item)
        return mat(temp_mat)
    else:
        raise Exception("矩阵A的列不等于矩阵B的行，乘不了的，小兄弟")

def dotk(k, matrix):
    '''常数乘矩阵'''
    if not (isinstance(k, float) or isinstance(k, int)) and \
            (isinstance(matrix, mat) or isinstance(matrix, list)):
        raise TypeError("k不是常数或者matrix不是矩阵")
    matrix = mat(matrix)
    for i in range(matrix.row):
        for j in range(matrix.col):
            matrix.mat[i][j] = k * matrix.mat[i][j]
    return mat(matrix.mat)

def fillMat(shape, fill = 0):
    i, j = shape
    return mat([[fill] * j for i in range(i)])

def column_stack(matrix, add_mat):
    '''
    将矩阵和矩阵进行组合，列组合，目前只能实现添加到最后面
    :param matrix: 原矩阵
    :param add_mat: 补充的列较少的矩阵
    :return: mat类型的矩阵
    '''
    m = copy.deepcopy(matrix)
    add = copy.deepcopy(add_mat)
    if isinstance(matrix, mat) and isinstance(add_mat, mat):
        n = len(matrix.mat)
        for i in range(n):
            m.mat[i].extend(add.mat[i])
        return mat(m)
    elif isinstance(add_mat, list) and isinstance(add_mat, list):
        n = len(matrix)
        for i in range(n):
            m[i].extend(add[i])
        return mat(m)
    else:
        raise TypeError("吔屎啦你，注意类型！")

def row_stack(matrix, add_mat):
    '''
    将矩阵和矩阵进行组合，行组合
    :param matrix: 原矩阵
    :param add_mat: 补充的列较少的矩阵
    :return: mat类型的矩阵
    '''
    m = mat(copy.deepcopy(matrix))
    add = mat(copy.deepcopy(add_mat))
    if (isinstance(matrix, mat) or isinstance(matrix, list))  \
        and (isinstance(add_mat, mat) or isinstance(add_mat, list)):
        if m.col == add.col:
            m.mat.extend(add.mat)
            return m
        else:
            raise ValueError("列数不一样！！！")
    else:
        raise TypeError("吔屎啦你，注意类型！")

def swap(matrix, i, j, RowOrCol):
    '''
    矩阵交换行
    :param mat:
    :param i:
    :param j:
    :return:
    '''
    if not isinstance(matrix, list):
        raise TypeError("交换行/列操作的必须时列表口牙！")
    else:
        x = copy.deepcopy(matrix)
        if RowOrCol == 0:
            x[i], x[j] = x[j], x[i]
        elif RowOrCol == 1:
            x = map(list, zip(*x))
            x[i], x[j] = x[j], x[i]
            x = map(list, zip(*x))
        return x

def uptriangulation(matrix):
    '''初等变换'''
    m_chushi = copy.deepcopy(matrix)
    row = len(m_chushi)
    col = len(m_chushi[0])
    # tflag为转置标志，为0时，原矩阵未转置，唯1时原矩阵被转置，输出时需要变回来
    tflag = 0
    if row >= col:
        m = m_chushi
    else:
        m = mat(m_chushi).T.mat
        row = len(m_chushi[0])
        col = len(m_chushi)
        tflag = 1
    # 以行大于等于列的情况下初等变换
    # 首先找出全为0的列
    colallzero = []
    for j in range(col):
        i = 0
        while i < row:
            if m[i][j] != 0:
                break
            i += 1
        if i == row:
            colallzero.append[j]
    # 全为0的列初等变换移到最后，越靠前的越靠后
    if len(colallzero) == col:
        print("0矩阵变换你妈")
    else:
        i = 0
        for j in range(col - 1, col - len(colallzero) - 1, -1):
            if j in colallzero:
                continue
            else:
                m = swap(m, colallzero[i], j, 1)
                i += 1
    # 换列后初等变换
    for j in range(col - len(colallzero)):
        k = j
        # 找到每一列第一个不为0的值
        while k < row:
            if m[k][j] != 0:
                break
            k += 1
        if k != row:
            rowchange = m[k]
            m[k] = m[j]
            m[j] = rowchange
            for i in range(j + 1, row):
                elfac = float(m[i][j] / m[j][j])
                for l in range(j, col - len(colallzero)):
                    m[i][l] = m[i][l] - elfac * m[j][l]
        else:
            continue

    if tflag == 0:
        return m
    else:
        return mat(m).T.mat

def MeanAndVar(x):
    '''返回矩阵每列均值和方差列表的元组'''
    if not isinstance(x, mat):
        raise TypeError("类型错误，我都嫌累了")
    m = mat(copy.deepcopy(x)).T
    mean = [i/float(m.col) for i in list(map(sum, m.mat))]
    var = map(lambda x:x/float(m.col), [sum(map(lambda x:(x-mean[i])**2, m.mat[i]))
           for i in range(m.row)])
    # return mean, var、
    return list(zip(mean, var))




'''
以下为测试矩阵，可以试试
'''

if __name__ == '__main__':

    x = [[3, 2, 6, 1, 3, 6, 1, 5, 11, 0],
            [1, 5, 7, 0, 1, 0, 1, 65, 2, 9], [1, 4, 6, 2, 11, 0, 1, 5, 7, 0], [1, 2, 6, 12, 42, 0, -4, 1, 11, 0],
            [1, 1, 2, 2, 23, 0, -14, 1, 1, 0],
            [23, 2, 3, 1, 44, 5, 1, -5, 0, 5], [1, 4, 6, 1, 4, 1, 67, -12, 1, 4], [1, 2, 6, 12, 42, 0, -4, 2, 11, 0],
            [1, 3, 16, 12, 42, 0, -4, 2, 11, 0],
            [1, 2, 6, 12, 4, 3, -4, 1, 11, 0]]

    a = [[1, 2], [1, 4], [3, 2], [25, 10]]
    b1 = [[3, 6, 9], [2, 3, 4], [1, 3, 6]]
    b2 = [[3, 6, 9, 3, 1], [2, 3, 4, 1, 1], [2, 1, 2, 5, 2], [2, 3, 1, 8, 4], [3, 6, 9, 3, 8]]
    c = [[0.5, 1, 0], [2, 1.5, 1], [0.2, 1, 2.5]]
    d = [[2, 5, -6], [4, 13, -19], [-6, -3, -6], [2, 1, 2]]
    time_start = time.clock()
    # print(swap(b1, 0, 1, 1))
    # a = mat(x).I.T
    # print(a)
    # print(dot(a, x))
    # print(mat(x).det)
    # test = random.rand((150, 150))
    time_end = time.clock()
    print('totally cost:', time_end - time_start)



