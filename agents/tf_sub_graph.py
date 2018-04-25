# -*- coding: utf-8 -*-
# @Time    : 2018/2/12 上午12:38
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : tf_sub_graph.py
# @Software: PyCharm Community Edition


import tensorflow as tf


class TFSubGraph(object):
    def __init__(self, scope, inputs):
        self.scope = scope
        self.inputs = inputs
        self.outputs = {}

    def define_graph(self):
        with tf.variable_scope(self.scope):
            self.create_variables()
            self.implement_graph()

    def create_variables(self):
        pass

    def implement_graph(self):
        pass

