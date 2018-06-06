#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 15:09
# @Author  : Hanwei Zhu
# @File    : agent.py

import os
import random
from collections import deque
import numpy as np
import tensorflow as tf
import constants as const
from agents import q_network, drq_network
import pandas as pd
from sklearn.preprocessing import Normalizer


class BaseAgent(object):
    def __init__(self, name, epsilon=const.EPSILON_INIT, learning_rate=0.01, learning_mode=True):
        self._name = name
        self.epsilon = epsilon
        self.gamma = const.GAMMA
        self._learning_rate = learning_rate
        self._learning_mode = learning_mode
        self.memory = deque(maxlen=const.MEMORY_SIZE)
        self._input_x = tf.placeholder(tf.float32,
                                       shape=[None, const.STATE_SPACE])
        self._input_y = tf.placeholder(tf.float32,
                                       shape=[None, const.ACTION_SPACE])
        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                 shape=[],
                                                 name='dropout_keep_prob')
        self.saver = tf.train.Saver()
        np.random.seed(seed=hash(self._name) % 256)

        self._target_q = None
        self._online_q = None
        self.replace_target_op = None
        self.sess = None

    def choose_action(self, state):
        action = self.sess.run(self._online_q.outputs[const.MAX_Q_OUTPUT],
                               feed_dict={self._input_x: state})

        if not self._learning_mode or random.random() >= self.epsilon:
            return action
        else:
            return np.random.randint(const.ACTION_SPACE)

    def update_target_q(self):
        self.sess.run(self.replace_target_op)

    def load(self):
        self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)

    def save_model(self):
        if not os.path.exists(const.MODEL_SAVE_PATH):
            os.makedirs(const.MODEL_SAVE_PATH)
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)


class DqnAgent(BaseAgent):
    def __init__(self, name, epsilon=const.EPSILON_INIT, learning_rate=0.01, learning_mode=True):
        super().__init__(name, epsilon, learning_rate, learning_mode)
        self._target_q = q_network.TargetQNetwork(scope=const.TARGET_Q_SCOPE + name,
                                                  inputs=(self._input_x, self._input_y))
        self._online_q = q_network.OnlineQNetwork(scope=const.ONLINE_Q_SCOPE + name,
                                                  inputs=(self._input_x, self._input_y))

        self._target_q.define_graph()
        self._online_q.define_graph()

        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q')
        online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online_q')

        with tf.variable_scope('soft_replacement'):
            self.replace_target_op = [tf.assign(t, o) for t, o in zip(target_params, online_params)]

        self.sess = tf.Session()
        if os.path.exists(const.MODEL_SAVE_PATH):
            self.load()
            print("load successfully")
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def learn(self):
        # sample batch memory from all memory
        mini_batch = random.sample(self.memory, const.MINI_BATCH_SIZE)

        states = np.asarray([i[0] for i in mini_batch])
        states = states.reshape((const.MINI_BATCH_SIZE, const.STATE_SPACE))

        targets = np.zeros((const.MINI_BATCH_SIZE, const.ACTION_SPACE))

        for index, [state, action, reward, next_state, done] in enumerate(mini_batch):
            q_eval = self.sess.run(
                self._online_q.outputs[const.Q_VALUE_OUTPUT],
                feed_dict={
                    self._input_x: state,
                })

            q_next = self.sess.run(
                self._target_q.outputs[const.MAX_Q_OUTPUT],
                feed_dict={
                    self._input_x: next_state,
                })

            target = reward

            if not done:
                target += self.gamma * q_next

            target_f = q_eval
            target_f[0][action] = target
            targets[index] = np.reshape(target_f, const.ACTION_SPACE)

        _, loss = self.sess.run([self._online_q.outputs[const.ADAM_OPTIMIZER],
                                 self._online_q.outputs[const.REDUCE_MEAN_LOSS]],
                                feed_dict={self._input_x: states, self._input_y: targets})

        self.epsilon -= const.EPSILON_DECAY

    def store_experience(self, state, action, reward, next_state, done):
        transition = [state, action, np.asarray(reward), next_state, done]
        self.memory.append(transition)


class DrqnAgent(BaseAgent):
    def __init__(self, name, epsilon=const.EPSILON_INIT, learning_rate=0.01, learning_mode=True):
        super().__init__(name, epsilon, learning_rate, learning_mode)
        self._target_q = drq_network.TargetDRQNetwork(scope=const.TARGET_Q_SCOPE + name,
                                                      inputs=(self._input_x, self._input_y),
                                                      dropout_keep_prob=self._dropout_keep_prob)
        self._online_q = drq_network.OnlineDRQNetwork(scope=const.ONLINE_Q_SCOPE + name,
                                                      inputs=(self._input_x, self._input_y),
                                                      dropout_keep_prob=self._dropout_keep_prob)
        self._target_q.define_graph()
        self._online_q.define_graph()

        target_q_weights = self._target_q.layered_cell.trainable_variables
        online_q_weights = self._online_q.layered_cell.trainable_variables

        self.replace_target_op = \
            [tf.assign(t, o) for t, o in zip(target_q_weights, online_q_weights)]

        # The rl_brain should hold a tf session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        # train_writer = tf.summary.FileWriter(const.LOG_PATH, sess.graph)
        if os.path.exists(const.MODEL_SAVE_PATH):
            print("load successfully")
            self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
        np.random.seed(seed=hash(self._name) % 256)

    def choose_action(self, state):
        q_values = self.sess.run(self._online_q.outputs[const.Q_VALUE_OUTPUT],
                                 feed_dict={self._input_x: state,
                                            self._dropout_keep_prob: 0.5})

        if not self._learning_mode or random.random() >= self.epsilon:
            return np.argmax(q_values[:, -1, :], axis=1)
        else:
            return np.random.randint(const.ACTION_SPACE)

    def learn(self):
        # sample batch memory from all memory
        (states, actions, rewards, terminate) = random.sample(self.memory, 1)[0]
        episode_length = len(actions)

        # current states
        q_eval = self.sess.run(
            self._online_q.outputs[const.Q_VALUE_OUTPUT],
            feed_dict={
                self._input_x: states,
                self._dropout_keep_prob: 1
            })
        # next states
        q_next = self.sess.run(
            self._target_q.outputs[const.MAX_Q_OUTPUT],
            feed_dict={
                self._input_x: states[:, 1:, :],
                self._dropout_keep_prob: 1
            })

        targets = np.zeros(q_eval.shape)

        # Bootstrapped Sequential Updates
        for index in range(episode_length):

            target = rewards[index]
            if not terminate[index]:
                target = target + self.gamma * q_next[:, index]

            target_f = q_eval[:, index, :]
            target_f[0][actions[index]] = target
            targets[:, index, :] = target_f

        _, loss = self.sess.run([self._online_q.outputs[const.ADAM_OPTIMIZER],
                                 self._online_q.outputs[const.REDUCE_MEAN_LOSS]],
                                feed_dict={self._input_x: states,
                                           self._input_y: targets,
                                           self._dropout_keep_prob: 0.8})

        self.epsilon -= const.EPSILON_DECAY

    def store_experience(self, state, actions, rewards, terminate):
        transition = (state, actions, rewards, terminate)
        self.memory.append(transition)


if __name__ == "__main__":
    # formed data
    const.initialize(state_space=4, action_space=3, n_agents=1, weight=0.5)
    a = DqnAgent("DQN_test")
    ac = a.choose_action([[0, 0, 0, 0]])
    print(ac)
