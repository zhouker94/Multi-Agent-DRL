#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:15
# @Author  : Hanwei Zhu
# @File    : base_agent.py


import tensorflow as tf
import numpy as np
import os
import random
from abc import abstractmethod


class BaseAgent(object):
    def __init__(self, name, opt, learning_mode=True):
        self._name = name
        self.opt = opt
        self.epsilon = self.opt["init_epsilon"]
        self.gamma = self.opt["gamma"]
        self._learning_rate = self.opt["learning_rate"]
        self._learning_mode = learning_mode

        self.buffer = np.zeros((self.opt["memory_size"],
                                self.opt["state_space"] + 1 + 1 + self.opt["state_space"]))
        self.buffer_count = 0

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

        self.update_q_net = None
        self.q_values_predict = None

        self._build_model()
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # self.writer = tf.summary.FileWriter(self.opt["summary_path"] + self._name + '/', self.sess.graph)
        np.random.seed(seed=hash(self._name) % 256)

    def start(self, dir_path):
        """
        Start: new/load a session
        """
        if os.path.exists(dir_path + self._name):
            self.saver.restore(self.sess, dir_path + self._name)
            print("load successfully")
        else:
            self.sess.run(self.init_op)

    def choose_action(self, state):
        """
        Choose an action
        """
        if not self._learning_mode or random.random() >= self.epsilon:
            action = np.argmax(self.sess.run(self.q_values_predict,
                                             feed_dict={self._state: state}))
        else:
            action = np.random.randint(self.opt["action_space"])
        return action

    @abstractmethod
    def _build_model(self):
        """
        Build nn model. All subclass should override this function
        """
        pass

    def update_q(self):
        """
        Copy weights from eval_net to target_net
        """
        self.sess.run(self.update_q_net)

    def save(self, dir_path):
        """
        Save Tensorflow model
        """
        if not os.path.exists(dir_path + self._name):
            os.makedirs(dir_path + self._name)
        save_path = self.saver.save(self.sess, dir_path + self._name + '/' + self._name)
        print("Model saved in path: %s" % save_path)

    def save_transition(self, state, action, reward, state_next):
        """
        Save transition to buffer
        """
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_count % self.opt["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_count += 1
