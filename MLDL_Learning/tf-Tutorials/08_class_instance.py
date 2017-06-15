#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date    : 2017-06-12 17:46:30
@Author  : gengshuoqin (gengshuoqin@360.com)
description:
"""
import os

"""
    class后面紧接着是类名，类名通常是大写开头的单词，
    紧接着是(object)，表示该类是从哪个类继承下来的通常，如果没有合适的继承类，就使用object类，这是所有类最终都会继承的类。
class Class_name(object):
    bart=Student()
    定义好了Student类，就可以根据Student类创建出Student的实例，创建实例.
    由于类可以起到模板的作用，因此，可以在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。通过定义一个特殊的__init__方法，在创建实例的时候，就把name，score等属性绑上去：
    __init__方法的第一个参数:self，表示创建的实例本身，因此，在__init__方法内部，就可以把各种属性绑定到self，因为self就指向创建的实例本身。
def __init__(self, name, score):
    self.name = name
    self.score = score
"""

class Student(object):
    def __init__(self,name,score):
        self.name=name
        self.score=score
    def print_score(self):
        print '2222'
        print '%s; %s'%(self.score,self.score)

"""
有了__init__方法，在创建实例的时候，就不能传入空的参数了，必须传入与__init__方法匹配的参数，但self不需要传，Python解释器自己会把实例变量传进去：
"""
bart2=Student('instance_name', 60)
bart2.print_score()