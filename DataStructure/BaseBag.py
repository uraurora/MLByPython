#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 数据结构的基础包"""
__author__ = 'Sei Gao'
# date = 2019.2.13

from array import array

class BagInterface(object):

    '''interface for all bag types'''

    def __init__(self, sourceCollection = None):
        pass

    def isEmpty(self):
        return True

    def __len__(self):
        return 0

    def __str__(self):
        return ""

    def __iter__(self):
        return None

    def __add__(self, other):
        return None

    def __eq__(self, other):
        return False

    def clear(self):
        pass

    def add(self, item):
        pass



