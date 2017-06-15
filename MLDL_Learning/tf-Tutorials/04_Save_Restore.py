#! /usr/bin/env python
#coding=utf-8

"""
tf tutorials
01 cnn save and resore
:保存以及恢复神经网络中的变量
 earlay-stop :在优化的过程中，当验证集上分类准确率提高时，保存神经网络的变量。如果经过1000次迭代还不能提升性能时，就终止优化。然后我们重新载入在验证集上表现最好的变量。
"""


import tensorflow as tf
import numpy as np

import os
# load data
# data dimensions
# model hyperparameters
# cnn model
# tf graph

# load data
from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets("data/MINIST",one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# data.test.cls = np.array([label.argmax() for label in data.test.labels])
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)
print("label in one hot index:\t{}".format(data.test.cls[0:5]))

# data dimensions
img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_classes=10
num_channels=1

# Model hyperparameters
filter_size1=5
num_filters1=16
filter_size2=5
num_filters2=32
fc_size=128


# Model
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def new_biases(num_filters):
	return tf.Variable(tf.constant(0.1,shape=[num_filters]))

def new_conv_layer(input,
				   num_input_channels,
				filter_size,
				num_filters,
				use_pooling):
	shape=[filter_size,filter_size,num_input_channels,num_filters]
	weights=new_weights(shape=shape)
	biases=new_biases(num_filters=num_filters)

	layer=tf.nn.conv2d(input=input,
					filter=weights,
					strides=[1,1,1,1],
					 padding='SAME')
	layer+=biases
	if use_pooling:
		layer=tf.nn.max_pool(value=layer,
							ksize=[1,2,2,1],
							strides=[1,2,2,1],
							padding='SAME')
	layer=tf.nn.relu(layer)

	return layer,weights
def flatten_layer(layer):
	layer_shape=layer.get_shape()
	num_features=layer_shape[1:4].num_elements()
	layer_flat=tf.reshape(layer, shape=[-1,num_features])
	return layer_flat,num_features

def new_fc_layer(input,
				num_inputs,
				num_outputs,
				use_relu=True):
	weights=new_weights(shape=[num_inputs,num_outputs])
	biases=new_biases(num_filters=num_outputs)
	layer=tf.matmul(input, weights)+biases
	if use_relu:
		layer=tf.nn.relu(layer)
	return layer

# tf Graph
x=tf.placeholder(dtype=tf.float32,shape=[None,img_size_flat],name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true=tf.placeholder(dtype=tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls=tf.argmax(y_pred,dimension=1)
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Training:
# run TF Graph
# #  Saver
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')
# Best validation accuracy seen so far.
best_validation_accuracy = 0.0
# Iteration-number for last improvement to validation accuracy.
last_improvement = 0
# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
batch_size = 256 # predict batch-size
def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.
    # The starting index for the next batch is denoted i.
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    return correct, cls_pred
def predict_cls_test():
    return predict_cls(images = data.test.images,
                       labels = data.test.labels,
                       cls_true = data.test.cls)
def predict_cls_validation():
    return predict_cls(images = data.validation.images,
                       labels = data.validation.labels,
                       cls_true = data.validation.cls)
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()

    # Calculate the classification accuracy and return it.
    return cls_accuracy(correct)
def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum
total_iterations = 0
train_batch_size = 64
def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement
    for i in range(num_iterations):
        total_iterations += 1
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        sess.run(optimizer, feed_dict=feed_dict_train)
        if total_iterations % 100 == 0 or (i == (num_iterations - 1)):
            acc_train=sess.run(accuracy,feed_dict=feed_dict_train)
            # msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            # print(msg.format(i + 1, acc))
            acc_validation, _ = validation_accuracy()
            # If validation accuracy is an improvement over best-known.
            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
            	# Save all variables of the TensorFlow graph to file
            	saver.save(sess=sess, save_path=save_path)
            	# A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
                # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))
        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")
            # Break out from the for-loop.
            break
def print_test_accuracy():
    correct, cls_pred = predict_cls_test()
    # acc, num_correct = cls_accuracy(correct)
    num_correct = correct.sum()
    acc = float(num_correct) / len(correct)
    # Number of images being classified.
    num_images = len(correct)
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
print_test_accuracy()
optimize(150)
print_test_accuracy()
# restore model
sess.run(init)
print_test_accuracy()
saver.restore(sess=sess, save_path=save_path)
print_test_accuracy()

