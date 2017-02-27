#! /usr/bin/python
# coding=utf-8
'''
Created on Feb 21, 2017
@author: gengsq2
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf





vocabulary_size = 50000
#a  list of a words
words=['geng','shuo','Extract', 'the ','first ','file' ,'enclosed','geng','geng'
       ,'geng','geng','geng','geng','geng','geng','geng','shuo','shuo','shuo'
       ,'shuo','shuo','shuo','shuo','shuo'];
# 计数器　collections.Counter
# most_common 返回一个TopN列表。如果n没有被指定，则返回所有元素。当多个元素计数值相同时，按照字母序排列。
count2 = collections.Counter(words).most_common(5)
count1 = [['UNK', -1]]
# print (count1)
# print (count2)
count1.extend(collections.Counter(words).most_common(8))
# print (count1)
dictionary = dict()
for word, _ in count1:
    dictionary[word]=len(dictionary)
# print ("dictionary[word]")
# print (dictionary[word])
data=list()
unk_count= 0
for word in words:
    if word in dictionary:
        index=dictionary[word]
    else:
        index=0
        unk_count +=1
    data.append(index)
count1[0][1] = unk_count 
dict_value=dictionary.values();
dict_key=dictionary.keys();
# print('dict_value')
# print(dict_value)
# print('dict_key')
# print(dict_key)
dict_zip=zip(dict_value,dict_key)
# print('dict_zip')
# print(dict_zip)
dict_=dict(dict_zip)
# print('dict_')
# print(dict_)
# 合使用zip( )和dict( )可以很方便的反转字典
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))   
# print('aaaa')
# print (reverse_dictionary)   
print('Most common words (+UNK)', count1[:2])
# data 是key  
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# global 声明全局变量
#　assert 断言

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  print ('len(data)')
  print (len(data))
  for _ in range(span):
    print (data_index)
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels
ba,ka=generate_batch(8, 2, 1)
print(ba)
print(ka)


# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2 
graph=tf.Graph()

with graph.as_default():
    
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
#     valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    with tf.device('/cpu:0'):
        # look up embeddings for inouts 
        embeddings =tf.Variable(
            tf.random_uniform([vocabulary_size,embedding_size], -1.0, 1.0))
        # 根据train_inputs中的id，寻找embeddings中的对应元素
        embed=tf.nn.embedding_lookup(embeddings,train_inputs)