#! /usr/bin/env python
#coding=utf-8

"""
tf tutorials
01 simple linear models

"""
import tensorflow as tf
import numpy as np

# load Data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
"""
load data labels as One-Hot encodeing;
the labels have been converted from a single number to a vector whose
length equals the number of possible classes.

"""
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print("label in one hot index:\t{}".format(data.test.cls[0:5]))
# data dimensions
# mnist images are 28pixels X 28pixels
img_size=28
# image stored as one-dimensional arrays
img_size_flat=img_size*img_size
# tuple with hight and width
img_shape=(img_size,img_size)
# Number of classes, one class for each of 10 digits.
num_classes=10
# Tensorflow Graph
# placeholder variables: tf Graph input
"""
Placeholder variables serve as the input to the graph
that we may change each time we execute the graph.
Placeholder是作为图的输入，每次我们运行图的时候都可能会改变它们
"""
x=tf.placeholder(tf.float32,[None,img_size_flat])
y_true=tf.placeholder(tf.float32,[None,num_classes])
# ????
y_true_cls = tf.placeholder(tf.int64, [None])
# variables to be optimized:weigts and bias
# 第一个需要优化的变量称为权重weight，TensorFlow变量需要被初始化为零
weights=tf.Variable(tf.zeros([img_size_flat,num_classes]))
# 第二个需要优化的是偏差变量biases，
# 它被定义成一个长度为num_classes的1维张量（或向量）。
biases=tf.Variable(tf.zeros([num_classes]))

# model
"""
最基本的数学模型将placeholder变量x中的图像与权重weight相乘，
然后加上偏差biases
tf.matnul(x,weights):[num_images,num_classes]
logits:[images,num_classes]
第i行第j列的那个元素代表着第i张输入图像有多大可能性是第j个类别。
这是很粗略的估计并且很难解释，
因为数值可能很小或很大，因此我们想要对它们做归一化，
使得logits矩阵的每一行相加为1，每个元素限制在0到1之间。
这是用一个称为softmax的函数来计算的，结果保存在y_pred中。
"""
logits=tf.matmul(x,weights)+biases
y_pred=tf.nn.softmax(logits)
# 从y_pred矩阵中取每行最大元素的索引值，来得到预测的类别。
# dimension:0:按列取最大值的索引;
# 1:按行取最大值的索引
y_pred_cls=tf.argmax(y_pred,dimension=1)
# print (y_pred_cls)
"""
优化损失函数
optmized cost function
了使模型更好地对输入图像进行分类，我们必须改变weights和biases变量。
首先我们需要比较模型的预测输出y_pred和期望输出y_true，来了解目前模型的性能如何。
交叉熵（cross-entropy）是一个在分类中使用的性能度量。
交叉熵是一个常为正值的连续函数，如果模型的预测值精准地符合期望的输出，它就等于零。
因此，优化的目的就是最小化交叉熵，通过改变模型中weights和biases的值，
使交叉熵越接近零越好。
TensorFlow有一个内置的计算交叉熵的函数。需要注意的是它使用logits的值，
因为在它内部也计算了softmax。
四种交叉熵
sigmoid.cross_entropy for mult-classes
softmax.cross_entropy

"""
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
"""
calculate cost
现已为每个图像分类计算了交叉熵，所以有一个当前模型在每张图上的性能度量。
但是为了用交叉熵来指导模型变量的优化，我们需要一个额外的标量值，
因此我们简单地利用所有图像分类交叉熵的均值。

"""
cost=tf.reduce_mean(cross_entropy)
"""
optimization
minimized the cost,creat a optimizer
use gradient descent with step-size is set to 0.5
优化过程并不是在这里执行。
实际上，还没计算任何东西，我们只是往TensorFlow图中添加了优化器，以便之后的操作。
多种优化方法:
gradientDescentOptimizer,AdagradOptimizer 或 AdamOptimizer

"""
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
# optimizer=tf.train.AdagradOptimizer(learning_rate=0.5).minimize(cost)
"""
Performance measures
性能度量，来向用户展示这个过程
这是一个布尔值向量，代表预测类型是否等于每张图片的真实类型。
"""
correct_prediction=tf.equal(y_pred_cls, y_true_cls)
"""
计算分类的准确度:
tf.cast(x,type)  将布尔值向量类型x转换成浮点型向量，这样子False就变成0，True变成1
然后计算这些值的平均数，
"""
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
"""
Create TensorFlow session
TensorFlow会话
"""
session=tf.Session()
"""
初始化变量
在开始优化weights和biases变量之前对它们进行初始化。
"""
init=tf.global_variables_initializer()
session.run(init)
"""
利用随机梯度下降的方法，它在优化器的每次迭代里只用到了一小部分的图像。
"""
batch_size=100
"""
执行了多次的优化迭代来逐步地提升模型的weights和biases。在每次迭代中，
从训练集中选择一批新的数据，然后TensorFlow用这些训练样本来执行优化器。
"""
def optimize(num_iterations):
	for i in range(num_iterations):
		# Get a batch of training examples.
		x_batch,y_true_batch=data.train.next_batch(batch_size)
		# Put the batch into a dict
		feed_dict_train={x:x_batch,
						 y_true:y_true_batch}
		"""
		使用batch训练数据来运行optimizer
		session.run(optimizer,feed_dict=feed_dict_train)
		sess.run([optimizer, cost], feed_dict=feed_dict_train)
		"""
		session.run(optimizer,feed_dict=feed_dict_train)


feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
"""
优化之前的性能
测试集上的准确度是9.8%。
这是由于模型只做了初始化，并没做任何优化，所以它通常将图像预测成数字零
"""
print_accuracy()
"""
迭代优化后的性能
"""
optimize(num_iterations=100)
print_accuracy()
