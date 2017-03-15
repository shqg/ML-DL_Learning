#! /usr/bin/python
# coding=utf-8
'''
Created on Feb 27, 2017
@author: gengsq2
'''
import numpy as np

def softmax(x):
    if len(x.shape)>1:
        tmp=np.max(x, axis=1)
        x-=tmp.reshape((x.shape[0],1))
        x=np.exp(x)
#         print x
#         print "xxxxxxxxxx"
        tmp=np.sum(x,axis=1)
#         print tmp
#         print "xxxxxxxxxx"
        x/=tmp.reshape(x.shape[0],1)
    else:
        tmp=np.max(x)
        x -=tmp
        x=np.exp(x)
        tmp=np.sum(x)
        x /=tmp
    return x

def test_softmax_basic():
    
    test1=softmax(np.array([1,2]))
#     print test1
    
    assert np.amax(np.fabs(test1-np.array([0.26894142,0.73105858]))) <=1e-6
    
    test2=softmax(np.array([[1001,1002,3],[3,4,5]]))
    print test2

if __name__=="__main__":
    test_softmax_basic()