#! /usr/bin/python
# coding=utf-8
'''
Created on Feb 28, 2017
@author: gengsq2
'''
import numpy as np

def sigmoid(x):
    "compute sigmoid function"
    x=1./(1+np.exp(-x))
    return x

def sigmoid_grad(f):
    f=f*(1-f)
    return f

def test_sigmoid_basic():
    x=np.array([[1,2],[-1,-2]])
    f=sigmoid(x)
    g=sigmoid_grad(f)
#     print f
#     print "gg:"
#     print g
    
if __name__=="__main__":
    test_sigmoid_basic()