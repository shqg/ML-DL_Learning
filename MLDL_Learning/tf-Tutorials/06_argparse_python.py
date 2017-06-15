#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date    : 2017-06-07 18:59:42
@Author  : gengshuoqin (gengshuoqin@360.com)
description: Python命令行解析库argparse
python标准库推荐使用argparse模块对命令行进行解析。
"""
import os

import argparse
# 创建解析器
"""
class ArgumentParser (prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True)

创建一个ArgumentParser实例对象，ArgumentParser对象的参数都为关键字参数。
prog ：程序的名字，默认为sys.argv[0]，用来在help信息中描述程序的名称。
"""

parser = argparse.ArgumentParser()
"""
添加参数选项:
add_argument (name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
name or flags ：参数有两种，可选参数和位置参数。
nrgs： 参数的数量:input more parameter
值可以为整数N(N个)，*(任意多个)，+(一个或更多)
parser.add_argument('integers',  type=int, nargs='+',
        help='an integer for the accumulator')
>>  argparse_python.py 3 3 3
[3, 3, 3]



"""
parser.add_argument('integers',  type=int, nargs='+',
        help='an integer for the accumulator')
parser.add_argument('--foo',type=int)
parser.add_argument('-t','--test',type=int)
args = parser.parse_args()

print args.integers
print args.foo
print args.test
