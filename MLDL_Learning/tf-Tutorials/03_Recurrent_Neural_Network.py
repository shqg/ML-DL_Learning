#! /usr/bin/env python
#coding=utf-8
# from __future__ import print_function
"""
tf tutorials
03 Recurrent Neural Network (LSTM)
"""
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

import tensorflow as tf
from tensorflow.contrib import rnn
# load data
from tensorflow.examples.tutorials.mnist import input_data

# print("Size  :")
mnist = input_data.read_data_sets("data/MINIST/", one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))
# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10
# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    """
    tf.unstack:拆散tesnor
    array([[ 1,  2,  3,  3],
       [ 4,  5,  6,  6],
       [ 7,  8,  9,  9],
       [10, 11, 12, 12]], dtype=int32)
    >>> t4=tf.unstack(t3,4,1)
    >>> sess.run(t4)
    [array([ 1,  4,  7, 10], dtype=int32),
     array([ 2,  5,  8, 11], dtype=int32),
     array([ 3,  6,  9, 12], dtype=int32),
     array([ 3,  6,  9, 12], dtype=int32)]
    >>> t4=tf.unstack(t3,2,1)
    """
    x = tf.unstack(x, n_steps, 1)
    # Define a lstm cell with tensorflow
    """
    num_units 是指一个Cell中神经元的个数，并不是循环层的Cell个数。。
    """
    lstm_cell = rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0,state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
pred = RNN(x, weights, biases)
# Define loss and optimizer
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


