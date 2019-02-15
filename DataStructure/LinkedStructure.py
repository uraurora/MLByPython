#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 链表结构的基础包"""
__author__ = 'Sei Gao'
# date = 2019.2.13


class Node(object):
    '''一个简单的节点类'''
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

class TwoWayNode(Node):
    '''双向节点，包含双向引用'''
    def __init__(self, data, previous=None, next=None):
        super(TwoWayNode, self).__init__(data, next)
        self.previous = previous

class LinkedList(object):
    node = {
        "1": Node,
        "2": TwoWayNode,
    }

    '''链表结构'''
    def __init__(self, type=1, length=10, filldata=None):
        if type == 1:
            self.type = "1"
            self.head = None
            for i in range(length):
                self.head = Node(i, self.head)
        elif type == 2:
            self.type = "2"
            self.head = TwoWayNode(None)
            self.tail = self.head
            for i in range(1, length):
                self.tail.next = TwoWayNode(i, self.tail)
                self.tail = self.tail.next
        else:
            raise ValueError("节点类型错误，填1/单向或2/双向")

    def __iter__(self):
        probe = self.head
        while probe != None:
            yield probe.data
            probe = probe.next

    def __len__(self):
        probe = self.head
        cursor = 0
        while probe != None:
            probe = probe.next
            cursor += 1
        return cursor

    def __index__(self, index):
        probe = self.head
        while index > 0:
            probe = probe.next
            index -= 1
        return probe.data

    def pop(self):
        '''删除最后一个节点的数据，返回其节点数据值'''
        if self.type == "1":
            probe = self.head
            while probe.next.next != None:
                probe = probe.next
            removedata = probe.next.data
            probe.next = None
        elif self.type == "2":
            removedata = self.tail.data
            self.tail = self.tail.previous
            self.tail.next = None
        return removedata



    def replace(self, index, newdata):
        '''替换指定位置的data值'''
        probe = self.head
        while index > 0:
            probe = probe.next
            index -= 1
        probe.data = newdata

    def insert(self, index, newdata):
        '''在指定位置插入某值'''
        if self.head == None or index <= 0:
            if self.type == "1":
                self.head = self.node[self.type](newdata, self.head)
            elif self.type == "2":
                self.head.next = self.node[self.type](newdata, self.head, self.head.next)
        else:
            if self.type == "1":
                probe = self.head
                while probe.next != None and index > 1:
                    probe = probe.next
                    index -= 1
                probe.next = self.node[self.type](newdata, probe.next)
            elif self.type == "2":
                probe = self.head
                while probe.next != None and index > 1:
                    probe = probe.next
                    index -= 1
                probe.next = self.node[self.type](newdata, probe, probe.next)

    def tolist(self):
        targetList = list()
        for i in self:
            targetList.append(i)
        return targetList

class CircularList(object):
    def __init__(self):
        pass






if __name__ == '__main__':
    # head = None
    # for count in range(1, 6):
    #     head = Node(count, head)
    # while head != None:
    #     print head.data
    #     head = head.next
    #
    # head = TwoWayNode(1)
    # tail = head
    #
    # for data in range(2, 6):
    #     tail.next = TwoWayNode(data, tail)
    #     tail = tail.next
    # probe = tail
    # while probe != None:
    #     print probe.data
    #     probe = probe.previous
    a = LinkedList(type=2)
    for i in a:
        print i