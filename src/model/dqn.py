#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 15:09
# @Author  : Hanwei Zhu
# @File    : dqn.py

import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
from model import base_model
import random
import os


class DQNModel(base_model.BaseModel):
    def __init__(self, aid, config, ckpt_path):
        super().__init__(aid, config, ckpt_path)

        self.buffer = np.zeros(
            (self.config["memory_size"],
             self.config["state_space"] + 1 + 1 + self.config["state_space"])
        )
        self.buffer_count = 0

    def _build_graph(self):
        self._action = tf.placeholder(tf.int32, [None, ], name='input_action')
        self._reward = tf.placeholder(tf.float32, [None, ], name='input_reward')
        # w, b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net_' + self.model_id):
            # batch_norm_state = tf.contrib.layers.batch_norm(self._state)
            with tf.variable_scope('phi_net'):
                phi_state_layer_1 = tf.layers.dense(
                    self._state,
                    self.config["fully_connected_layer_1_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_layer_2 = tf.layers.dense(
                    phi_state_layer_1,
                    self.config["fully_connected_layer_2_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_layer_3 = tf.layers.dense(
                    phi_state_layer_2,
                    self.config["fully_connected_layer_3_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu)
                phi_state_layer_4 = tf.layers.dense(
                    phi_state_layer_3,
                    self.config["fully_connected_layer_4_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu)

            self.q_values_predict = tf.layers.dense(
                phi_state_layer_4,
                self.config["action_space"],
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='Q_predict')

            # tf.summary.histogram('q_values_predict', self.q_values_predict)

            with tf.variable_scope('q_predict'):
                # size of q_value_predict is [BATCH_SIZE, 1]
                action_indices = tf.stack(
                    [
                        tf.range(tf.shape(self._action)[0], dtype=tf.int32),
                        self._action
                    ],
                    axis=1
                )
                self.q_value_predict = tf.gather_nd(
                    self.q_values_predict,
                    action_indices
                )
                self.action_output = tf.argmax(self.q_values_predict)

        with tf.variable_scope('target_net_' + self.model_id):
            # batch_norm_next_state = tf.contrib.layers.batch_norm(self._next_state)

            with tf.variable_scope('phi_net'):
                phi_state_next_layer_1 = tf.layers.dense(
                    self._next_state,
                    self.config["fully_connected_layer_1_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_next_layer_2 = tf.layers.dense(
                    phi_state_next_layer_1,
                    self.config["fully_connected_layer_2_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_next_layer_3 = tf.layers.dense(
                    phi_state_next_layer_2,
                    self.config["fully_connected_layer_3_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_next_layer_4 = tf.layers.dense(
                    phi_state_next_layer_3,
                    self.config["fully_connected_layer_4_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )

            self.q_values_target = tf.layers.dense(
                phi_state_next_layer_4,
                self.config["action_space"],
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='Q_target'
            )

            with tf.variable_scope('q_real'):
                # size of q_value_real is [BATCH_SIZE, 1]
                q_value_max = tf.reduce_max(self.q_values_target, axis=1)
                q_value_real = self._reward + \
                    self.config["gamma"] * q_value_max
                self.q_value_real = tf.stop_gradient(q_value_real)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(
                    self.q_value_real,
                    self.q_value_predict,
                    name='mse'
                )
            )

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                self.config["learning_rate"]).minimize(self.loss)

        target_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope="target_net_" + self.model_id
        )
        eval_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope="eval_net_" + self.model_id
        )

        with tf.variable_scope('soft_replacement'):
            self.update_q_net = [
                tf.assign(t, e)
                for t, e in zip(target_params, eval_params)
            ]

    def fit(self):
        # sample batch memory from all memory
        if self.buffer_count > self.config["memory_size"]:
            sample_indices = np.random.choice(
                self.config["memory_size"],
                size=self.config["batch_size"]
            )
        else:
            sample_indices = np.random.choice(
                self.buffer_count,
                size=self.config["batch_size"]
            )

        batch = self.buffer[sample_indices, :]
        batch_s = batch[:, :self.config["state_space"]]
        batch_a = batch[:, self.config["state_space"]]
        batch_r = batch[:, self.config["state_space"] + 1]
        batch_s_n = batch[:, -self.config["state_space"]:]

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self._state: batch_s,
                self._action: batch_a,
                self._reward: batch_r,
                self._next_state: batch_s_n
            }
        )

        # self.writer.add_summary(summaries, global_step)

        if not self.step_counter % 100:
            self.update_q()

    def save_transition(self, state, action, reward, state_next):
        """
        Save transition to buffer
        """
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_count % self.config["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_count += 1

    def update_q(self):
        """
        Copy weights from eval_net to target_net
        """
        self.sess.run(self.update_q_net)

    def predict(self, state, epsilon, **kwargs):
        """
        Choose an action
        """
        if random.random() >= epsilon:
            action_idx = np.argmax(
                self.sess.run(
                    self.action_output,
                    feed_dict={self._state: state}
                )
            )
        else:
            action_idx = np.random.randint(self.config["action_space"])

        action = kwargs["pre_action"]
        if action_idx == 0:
            action += self.config["delta_increment"]
        elif action_idx == 1:
            action -= self.config["delta_increment"]
        return action
