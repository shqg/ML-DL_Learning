#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date    : 2017-06-12 11:00:40
@Author  : gengshuoqin (gengshuoqin@360.com)
description:
dense_to_one_hot:Convert class labels from scalars to one-hot vectors
"""
import os
import numpy as np
###################################################
"""
将 vector型的label转为oneHot
input:
labels_dense:   训练数据的第一列是label列;labels_dense=trianData[:,0]
num_classes: 分类数
output: oneHot label matrix 中的值为int型的
"""
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # print (index_offset + labels_dense.ravel()).astype(int)
    index_to_int=(index_offset + labels_dense.ravel()).astype(int)
    labels_one_hot.flat[index_to_int] = 1
    # print "labels_one_hot"
    # print labels_one_hot
    return labels_one_hot
# 07_sample.txt 第一列标签列{0,1}两类；07_sample02.txt 第一列标签列{0,1,2,3,4,5}六类
filename="/home/gsq/program/tf-Tutorials/data/07_sample.txt"
"""
调用read()会一次性读取文件的全部内容，所以，要保险起见，可以反复调用read(size)方法，每次最多读取size个字节的内容。另外，调用readline()可以每次读取一行内容，调用readlines()一次读取所有内容并按行返回list。因此，要根据需要决定怎么调用。
"""
with open(filename) as f:
    lines=f.readlines()
    # print len(lines)
    # for  line in lines:
        # print line
    data_vec=np.array(lines)
    # print data_vec.shape
    a = np.loadtxt(filename)
    labels=a[:,0]
    dense_to_one_hot(labels,2)
##################################################




