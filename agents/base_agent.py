#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:15
# @Author  : Hanwei Zhu
# @File    : base_agent.py


import tensorflow as tf
import numpy as np
import os
from abc import abstractmethod


class BaseAgent(object):
    def __init__(self, name, opt, learning_mode=True):
        self._name = name
        self.opt = opt
        self.epsilon = self.opt["init_epsilon"]
        self.gamma = self.opt["gamma"]
        self._learning_rate = self.opt["learning_rate"]
        self._learning_mode = learning_mode

        self._state = tf.placeholder(tf.float32,
                                     shape=[None, self.opt["state_space"]],
                                     name='input_state')
        self._next_state = tf.placeholder(tf.float32,
                                          shape=[None, self.opt["state_space"]],
                                          name='input_next_state')
        self._reward = tf.placeholder(tf.float32, [None, ], name='input_reward')
        self._action = tf.placeholder(tf.int32, [None, ], name='input_action')
        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                 shape=[],
                                                 name='dropout_keep_prob')

        self.action_output = None
        self._build_model()
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # self.writer = tf.summary.FileWriter(self.opt["summary_path"] + self._name + '/', self.sess.graph)
        np.random.seed(seed=hash(self._name) % 256)

    def start(self, dir_path=""):
        """
        Start: new/load a session
        """
        if self._learning_mode:
            self.sess.run(self.init_op)
        else:
            dir_path += self._name + '/'
            if os.path.exists(dir_path):
                self.saver.restore(self.sess, dir_path + self._name)
                print("load successfully")
            else:
                print("no model exists")

    @abstractmethod
    def _build_model(self):
        """
        Build nn model. All subclass should override this function
        """
        pass

    @abstractmethod
    def learn(self, global_step):
        pass

    def save(self, dir_path):
        """
        Save Tensorflow model
        """
        if not os.path.exists(dir_path + self._name):
            os.makedirs(dir_path + self._name)
        save_path = self.saver.save(self.sess, dir_path + self._name + '/' + self._name)
        print("Model saved in path: %s" % save_path)
