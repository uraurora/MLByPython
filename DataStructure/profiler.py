#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 算法测试器，包含排序算法"""
__author__ = 'Sei Gao'
# date = 2018.10.30
import random
import time
import numpy as np
# import redis

class Profiler(object):
    # 对算法进行简单的复杂度测试
    def test(self, function, lyst=None, size=10,
             unique=True, comp=True, exch=True, trace=False):
        '''
        :param function: 测试的算法
        :param lyst: 列表
        :param size: 列表长度吗，默认为10
        :param unique: True则，列表包含唯一的数字
        :param comp:True则计数比较
        :param exch:True则计数交换
        :param trace:True则打印列表每次变化
        :return:
        '''
        self._comp = comp
        self._exch = exch
        self._trace = trace
        if lyst != None:
            self._lyst = lyst
        elif unique:
            self._lyst = range(1, size+1)
            random.shuffle(self._lyst)
        else:
            self._lyst = []
            for count in range(size):
                self._lyst.append(random.randint(1, size))
        self._exchCount = 0
        self._cmpCount = 0
        self._startClock()
        function(self._lyst, self)
        self._stopClock()
        print(self)

    def exchange(self):
        if self._exch:
            self._exchCount += 1
        if self._trace:
            print(self._lyst)
    def comparison(self):
        if self._comp:
            self._cmpCount += 1

    def _startClock(self):
        self._start = time.time()
    def _stopClock(self):
        self._elapsedTime = round(time.time()-self._start, 5)

    def __str__(self):
        result = "Problem size:  "
        result += str(len(self._lyst)) + "\n"
        result += "Elapased time:  "
        result += str(self._elapsedTime) + "\n"
        if self._comp:
            result += "Comparisons:  "
            result += str(self._cmpCount) + "\n"
        if self._exch:
            result += "Exchanges:  "
            result += str(self._exchCount) + "\n"
        return result

class Sort:
    # 带有@classmethod的直接调用即可
    def __str__(self):
        return "排序算法类哦~"

    def swap(self, a, i, j, profiler):
        profiler.exchange()
        temp = a[i]
        a[i] = a[j]
        a[j] = temp

    @classmethod
    def bubbleSort(clf, array, profiler):
        for i in range(len(array)):
            for j in range(len(array)-i-1):
                if array[j] > array[j+1]:
                    profiler.comparison()
                    clf().swap(array, j, j+1, profiler)
        return array

    @classmethod
    def selectSort(clf, lyst, profiler):
        for i in range(len(lyst)):
            min = lyst[i]
            for j in range(i, len(lyst)):
                if min > lyst[j]:
                    profiler.comparison()
                    min = lyst[j]
                    clf().swap(lyst, i, j, profiler)

    @classmethod
    def quickSort(clf, lyst, profiler):
        clf().quickSortHelper(lyst, 0, len(lyst)-1, profiler)
        return lyst

    def quickSortHelper(self, lyst, left, right, profiler):
        if left < right:
            profiler.comparison()
            pivotLocation = self.partition(lyst, left, right, profiler)
            self.quickSortHelper(lyst, left, pivotLocation-1, profiler)
            self.quickSortHelper(lyst, pivotLocation+1, right, profiler)

    def partition(self, lyst, left, right, profiler):
        middle = (left+right)//2
        pivot = lyst[middle]
        lyst[middle] = lyst[right]
        lyst[right] = pivot

        boundary = left
        for index in range(left, right):
            if lyst[index] < pivot:
                profiler.comparison()
                self.swap(lyst, index, boundary, profiler)
                boundary += 1
        self.swap(lyst, right, boundary, profiler)
        return boundary

# 最大子列和问题
def MCS(lyst):
    max = 0
    sum = 0
    for i in lyst:
        sum += i
        if sum > max:
            max = sum
        elif sum < 0:
            sum = 0
    return max

if __name__ == '__main__':
    # 调用方式，实例化测试器类
    p = Profiler()
    # 选择不同的排序算法进行比较
    p.test(Sort().quickSort, size=5000, unique=False)
    p.test(Sort().selectSort, size=5000, unique=False)
    a = np.random.randint(1, 100, 100)
    # a = [1, 2, -4, 3, -2, 5, 2, -4, 1, -3, 2, 5, -5]
    # print(MCS(a))

