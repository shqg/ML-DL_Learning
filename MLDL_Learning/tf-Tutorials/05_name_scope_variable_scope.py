#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date    : 2017-06-05 02:52:39
@Author  : gengshuoqin (gengshuoqin@360.com)

description:
1. name_scope
2.variable_scope
3.get_variable_scope
4.get_variable()
5.tf.app.flags
"""
import tensorflow as tf
import numpy as np
"""
tf.name_scope:命名空间技术以避免冲突
有两种创建variable的方法：
tf.Variable()
tf.get_variable()：从同一个变量范围内获取或者创建，可根据 name 值，
返回该变量，如果该 name 不存在的话，则会进行创建；
创建的变量名不受 name_scope 的影响
tf.get_variable() 与 tf.Variable() 相比，多了一个 initilizer （初始化子）可选参数；
tf.Variable() 对应地多了一个 initial_value 关键字参数，也即对于 tf.Variable 创建变量的方式，必须显式初始化
"""
with tf.name_scope("name_scope"):
    init=tf.constant_initializer(value=1)
    var1=tf.get_variable(name='var1',shape=[1],dtype=tf.float32,initializer=init)
    var2=tf.Variable(name='var2',initial_value=2,dtype=tf.float32)
    var2_1=tf.Variable(name='var2',initial_value=2,dtype=tf.float32)
"""
   print:
    var1.name  var1:0  创建的变量名不受name_scope 影响
    var2.name name_scope/var2:0
    var2_1.name name_scope/var2_1:0 虽然var2和var2_1的name相同但输出是不同的，相当于创建了两个变量
 """
#重复利用之前创造的变量： 从上边的例子知道tf.Variable无法重用，即使同名也相当于重新创建
with tf.variable_scope('variable_scope') as scope:
	init=tf.constant_initializer(value=3)
	var3=tf.get_variable(name='var3',shape=[1],dtype=tf.float32,initializer=init)
	var4=tf.Variable(name='var4',initial_value=4,dtype=tf.float32)
	scope.reuse_variables()
	var5=tf.get_variable(name='var3') #重复使用variable
	"""
	print :
	var3.name: variable_scope/var3:0
	var5.name: variable_scope/var3:0
	是同一变量，实现重用
	"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)
    print(var2.name)
    print(var2_1.name)
    print(var3.name)
    print(var5.name)
    print(sess.run(var1))

"""
tf.app.flags，它支持应用从命令行接受参数，可以用来指定集群配置等。
如果不传参数，可以使用默认参数

#调用flags内部的DEFINE_string函数来制定解析规则
flags.DEFINE_string("para_name_1","default_val", "description")
flags.DEFINE_bool("para_name_2","default_val", "description")
#FLAGS是一个对象，保存了解析后的命令行参数
FLAGS = flags.FLAGS
def main(_):
    FLAGS.para_name #调用命令行输入的参数

if __name__ = "__main__": #使用这种方式保证了，如果此文件被其它文件import的时候，不会执行main中的代码
    tf.app.run() #解析命令行参数，调用main函数 main(sys.argv)
调用方法：
~/ python script.py --para_name_1 name --para_name_2 name2
# 不传的话，会使用默认值
"""
flags =tf.app.flags
FLAGS = flags.FLAGS
print ("flags test.....")
flags.DEFINE_integer('num_hidden_layers', 3, 'number of hidden layers')
flags.DEFINE_bool("para_name_2",True, "description")

# mian(unused_argv)和mian(_) 效果相同
# def main(unused_argv):
"""
命令行输入时不能省略flag;只能用flag不能像argument flags或name --para_name_1 
"""
def main(_):
    print'sssFLAGS.num_hidden_layers','%s' % FLAGS.num_hidden_layers
    if FLAGS.para_name_2:
		print 'para_name_2: true'
    else:
        print 'para_name_2: false'

if __name__ == '__main__':
    # main()
    tf.app.run()

