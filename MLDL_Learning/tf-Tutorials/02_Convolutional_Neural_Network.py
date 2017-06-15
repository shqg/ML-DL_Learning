#! /usr/bin/env python
#coding=utf-8

"""
tf tutorials
01 convolutional_neural networks

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
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# CNN Model Hyperparameters
# CNN layer 1
"""
Convolution filters are 5 x 5 pixels.
There are 16 of these filters
"""
filter_size1=5
num_filters1=16
# CNN layer 2
filter_size2=5
num_filters2=36
# full-connected layer:Number of neurons in fully-connected layer.
fc_size=128
"""
cnn layer:
TensorFlow在计算图里创建了新的卷积层。
这里并没有执行什么计算，只是在TensorFlow图里添加了数学公式。
input :A 4-d tensor; output:  Has the same type as input. A 4-D tensor.
输入的是四维的张量:图像数量;每张图像的Y轴;每张图像的X轴;每张图像的通道数
输出是另外一个4通道的张量:图像数量，与输入相同;每张图像的Y轴。如果用到了2x2的池化，
是输入图像宽高的一半。
;每张图像的X轴。同上;卷积滤波生成的通道数。

用于生成随机数tensor的。尺寸是shape
random_normal: 正太分布随机数，均值mean,标准差stddev
truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
random_uniform:均匀分布随机数，范围为[minval,maxval]
weights随机初始化;
bias 初始为固定值;偏置初始化是给所有的偏置赋相同的值
"""
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,num_input_channels,
					filter_size,num_filters,use_pooling):
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	"""
	创建模型之前，我们先来创建权重和偏置。一般来说，初始化时应加入轻微噪声，
	来打破对称性，防止零梯度的问题。因为我们用的是 ReLU，所以用稍大于 0 的值来初
	始化偏置能够避免节点输出恒为 0 的问题（ dead neurons）。

	tf.truncated_normal(shape, mean, stddev) :
	shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，
	均值和标准差自己设定。这是一个截断的产生正太分布的函数，
	就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
	和一般的正太分布的产生随机数据比起来，
	这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
	"""
	weights=tf.Variable(tf.truncated_normal(shape, stddev=0.05))
	biases= tf.Variable(tf.constant(0.05, shape=[num_filters]))
	"""
	 Create the TensorFlow operation for convolution.

	 Note :
     The first and last stride must always be 1,
     because the first is for the image-number and
     the last is for the input-channel.
     But e.g. strides=[1, 2, 2, 1] would mean that the filter
     is moved 2 pixels across the x- and y-axis of the image.
     The padding is set to 'SAME' which means the input image
     is padded with zeroes so the size of the output is the same.
	"""
	"""
	tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
	除去name参数用以指定该操作的name，与方法有关的一共五个参数：
	第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
	第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
	第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
	第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
	第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
	结果返回一个Tensor，这个输出，就是我们常说的feature map
	"""
	layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
	"""
	Add the biases to the results of the convolution.
    A bias-value is added to each filter-channel.
	"""
	# 16 matrix after conv layer  then + 16 biases
	layer+=biases
	# use pooling layer
	if use_pooling:
		"""
		This is 2x2 max-pooling {ksize=[1, 2, 2, 1]}, which means that we
        consider 2x2 windows and select the largest value
        in each window. Then we move 2 pixels{strides=[1, 2, 2, 1]} to the next window.
		"""
		layer=tf.nn.max_pool(value=layer,
								ksize=[1,2,2,1],
								strides=[1,2,2,1],
								padding='SAME')
	"""
	Rectified Linear Unit (ReLU).
    It calculates max(x, 0) for each input pixel x.
    This adds some non-linearity to the formula and allows us
    to learn more complicated functions.
	"""
	layer=tf.nn.relu(layer)

 	"""
	Note that ReLU is normally executed before the pooling,
    but since relu(max_pool(x)) == max_pool(relu(x)) we can
    save 75% of the relu-operations by max-pooling first.
	"""
	return layer,weights

"""
转换层:
卷积层生成了4维的张量。我们会在卷积层之后添加一个全连接层，因此我们需要将这个4维的张量转换成可被全连接层使用的2维张量。
"""
def flatten_layer(layer):
	# get the shape of the input layer
	layer_shape=layer.get_shape()
	# The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # ????????????????
    # ?tf.TensorShape.num_elements()
    # Returns the total number of elements, or none for incomplete shapes.
    # The number of features is: img_height * img_width * 	num_channels
    # We can use a function from TensorFlow to calculate this. Returns the total number of elements, or none for incomplete shapes.
	num_features=layer_shape[1:4].num_elements()
    # 所以-1，就是缺省值，就是先以你们合适，到时总数除以你们几个的乘积，我该是几就是几。
	layer_flat=tf.reshape(layer,[-1,num_features])
	return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

"""
tf.sparse_to_dense
训练数据从sparse转为dense 和label转为one-hot
"""
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
# tf.argmax return A Tensor of type int64.
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
"""
cross_entropy Returns:
A 1-D Tensor of length batch_size of the same type as logits with the softmax cross entropy loss.
"""
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session = tf.Session()

session.run(tf.global_variables_initializer())
train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    # start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        print('1')
        print(i)
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    # end_time = time.time()

    # Difference between start and end-times.
    # time_dif = end_time - start_time

    # Print the time-usage.
    # print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    # Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # # Plot some examples of mis-classifications, if desired.
    # if show_example_errors:
    #     print("Example errors:")
    #     plot_example_errors(cls_pred=cls_pred, correct=correct)

    # # Plot the confusion matrix, if desired.
    # if show_confusion_matrix:
    #     print("Confusion Matrix:")
    #     plot_confusion_matrix(cls_pred=cls_pred)
print_test_accuracy()

optimize(num_iterations=100)
print_test_accuracy()
