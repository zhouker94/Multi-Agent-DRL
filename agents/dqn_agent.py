#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 15:09
# @Author  : Hanwei Zhu
# @File    : dqn_agent.py


import numpy as np
import tensorflow as tf
from agents import base_agent
import random


class DqnAgent(base_agent.BaseAgent):
    def __init__(self, name, opt, learning_mode=True):
        super().__init__(name, opt, learning_mode)
        self.buffer = np.zeros((self.opt["memory_size"],
                                self.opt["state_space"] + 1 + 1 + self.opt["state_space"]))
        self.buffer_count = 0

    def _build_model(self):
        # w, b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net_' + self._name):
            batch_norm_state = tf.layers.batch_normalization(self._state)
            layer_norm_state = tf.contrib.layers.layer_norm(batch_norm_state)

            with tf.variable_scope('phi_net'):
                phi_state_layer_1 = tf.layers.dense(layer_norm_state,
                                                    self.opt["fully_connected_layer_1_node_num"],
                                                    kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer,
                                                    activation=tf.nn.relu)
                phi_state_layer_2 = tf.layers.dense(phi_state_layer_1,
                                                    self.opt["fully_connected_layer_2_node_num"],
                                                    kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer,
                                                    activation=tf.nn.relu)
                phi_state_layer_3 = tf.layers.dense(phi_state_layer_2,
                                                    self.opt["fully_connected_layer_3_node_num"],
                                                    kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer,
                                                    activation=tf.nn.relu)

            self.q_values_predict = tf.layers.dense(phi_state_layer_3,
                                                    self.opt["action_space"],
                                                    kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer,
                                                    name='Q_predict')
            # tf.summary.histogram('q_values_predict', self.q_values_predict)

            with tf.variable_scope('q_predict'):
                # size of q_value_predict is [BATCH_SIZE, 1]
                action_indices = tf.stack([tf.range(tf.shape(self._action)[0], dtype=tf.int32), self._action], axis=1)
                self.q_value_predict = tf.gather_nd(self.q_values_predict, action_indices)
                self.action_output = tf.arg_max(self.q_value_predict)

        with tf.variable_scope('target_net_' + self._name):
            batch_norm_next_state = tf.layers.batch_normalization(self._next_state)
            layer_norm_next_state = tf.contrib.layers.layer_norm(batch_norm_next_state)

            with tf.variable_scope('phi_net'):
                phi_state_next_layer_1 = tf.layers.dense(layer_norm_next_state,
                                                         self.opt["fully_connected_layer_1_node_num"],
                                                         kernel_initializer=w_initializer,
                                                         bias_initializer=b_initializer,
                                                         activation=tf.nn.relu)
                phi_state_next_layer_2 = tf.layers.dense(phi_state_next_layer_1,
                                                         self.opt["fully_connected_layer_2_node_num"],
                                                         kernel_initializer=w_initializer,
                                                         bias_initializer=b_initializer,
                                                         activation=tf.nn.relu)
                phi_state_next_layer_3 = tf.layers.dense(phi_state_next_layer_2,
                                                         self.opt["fully_connected_layer_3_node_num"],
                                                         kernel_initializer=w_initializer,
                                                         bias_initializer=b_initializer,
                                                         activation=tf.nn.relu)

            self.q_values_target = tf.layers.dense(phi_state_next_layer_3,
                                                   self.opt["action_space"],
                                                   kernel_initializer=w_initializer,
                                                   bias_initializer=b_initializer,
                                                   name='Q_target')

            with tf.variable_scope('q_real'):
                # size of q_value_real is [BATCH_SIZE, 1]
                q_value_max = tf.reduce_max(self.q_values_target, axis=1)
                q_value_real = self._reward + self.gamma * q_value_max
                self.q_value_real = tf.stop_gradient(q_value_real)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value_real, self.q_value_predict, name='mse'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)

        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net_" + self._name)
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net_" + self._name)

        with tf.variable_scope('soft_replacement'):
            self.update_q_net = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

    def learn(self, global_step):
        # sample batch memory from all memory
        if self.buffer_count > self.opt["memory_size"]:
            sample_indices = np.random.choice(self.opt["memory_size"], size=self.opt["batch_size"])
        else:
            sample_indices = np.random.choice(self.buffer_count, size=self.opt["batch_size"])

        batch = self.buffer[sample_indices, :]
        batch_s = batch[:, :self.opt["state_space"]]
        batch_a = batch[:, self.opt["state_space"]]
        batch_r = batch[:, self.opt["state_space"] + 1]
        batch_s_n = batch[:, -self.opt["state_space"]:]

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self._state: batch_s,
                self._action: batch_a,
                self._reward: batch_r,
                self._next_state: batch_s_n
            }
        )

        self.epsilon -= self.opt["epsilon_decay"]
        # self.writer.add_summary(summaries, global_step)

        if not global_step % 100:
            self.update_q()

    def save_transition(self, state, action, reward, state_next):
        """
        Save transition to buffer
        """
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_count % self.opt["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_count += 1

    def update_q(self):
        """
        Copy weights from eval_net to target_net
        """
        self.sess.run(self.update_q_net)

    def choose_action(self, state):
        """
        Choose an action
        """
        if not self._learning_mode or random.random() >= self.epsilon:
            action = np.argmax(self.sess.run(self.action_output,
                                             feed_dict={self._state: state}))
        else:
            action = np.random.randint(self.opt["action_space"])
        return action
