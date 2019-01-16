#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 画图的工具"""
__author__ = 'Sei Gao'
# date = 2019.01.14

try:
    import matplotlib.pyplot as plt
except:
    raise Exception("无法加载matplotlib库，看不了图了")

def ErrorPlot(error, length, a=0.5, color='red'):
    figure = plt.figure()
    plt.plot(error, alpha=a, c=color)
    plt.xlim(0, length)
    plt.xlabel("Iteration-Count")
    plt.ylabel("Error")
    figure.show()

