# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 下午8:47
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : utils.py
# @Software: PyCharm Community Edition

import tensorflow as tf


class Utils(object):
    @staticmethod
    def create_variable(name, scope, shape, dtype, initializer):
        var = None
        with tf.variable_scope(scope):
            var = tf.get_variable(name,
                                  shape=shape,
                                  dtype=dtype,
                                  initializer=initializer)

        return var

    @staticmethod
    def create_truncated_normal_variable(name,
                                         scope,
                                         shape,
                                         dtype=tf.float32,
                                         normal_mean=0.0,
                                         std_dev=0.1):
        initializer = tf.truncated_normal(shape=shape,
                                          mean=normal_mean,
                                          stddev=std_dev)

        return Utils.create_variable(name, scope, shape, dtype, initializer)

    @staticmethod
    def create_random_normal_variable(name,
                                      scope,
                                      shape,
                                      dtype=tf.float32,
                                      normal_mean=0.0,
                                      std_dev=1.0):
        initializer = tf.random_normal_initializer(normal_mean,
                                                   std_dev)
        return Utils.create_variable(name, scope, shape, dtype, initializer)

