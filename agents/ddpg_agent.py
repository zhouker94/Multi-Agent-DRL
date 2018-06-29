# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 下午3:48
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : ddpg_agent.py
# @Software: PyCharm Community Edition


from agents import base_agent
import tensorflow as tf
import numpy as np


class DDPGAgent(base_agent.BaseAgent):
    def __init__(self, name, opt, learning_mode=True):
        super().__init__(name, opt, learning_mode)
        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []

    def _build_model(self):
        # w, b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('policy_network'):
            batch_norm_state = tf.layers.batch_normalization(self._state)
            layer_norm_state = tf.contrib.layers.layer_norm(batch_norm_state)

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

        with tf.variable_scope('output'):
            self.action_output = tf.layers.dense(phi_state_layer_3,
                                                 self.opt["action_space"],
                                                 kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer,
                                                 activation=tf.nn.relu)

        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.action_output, self._action) * self._reward

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)

    def save_transition(self, state, action, reward):
        self.s_buffer.append(state)
        self.a_buffer.append(action)
        self.r_buffer.append(reward)

    def _get_normalized_rewards(self):
        reward_normalized = np.zeros_like(self.r_buffer)
        reward_delta = 0
        for index in reversed(range(0, len(self.r_buffer))):
            reward_delta = reward_delta * self.gamma + self.r_buffer[index]
            reward_normalized[index] = reward_delta
        reward_normalized -= np.mean(reward_normalized)
        reward_normalized /= np.std(reward_normalized)
        return reward_normalized

    def learn(self, global_step):
        reward_normalized = self._get_normalized_rewards()

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self._state: np.vstack(self.s_buffer),
            self._action: np.array(self.a_buffer),
            self._reward: reward_normalized,
        })

        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []
