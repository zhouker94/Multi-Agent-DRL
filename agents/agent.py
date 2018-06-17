#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 15:09
# @Author  : Hanwei Zhu
# @File    : agent.py

import os
import random
import numpy as np
import tensorflow as tf


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

        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()

        self.update_q_net = None
        self.q_values_predict = None

        self._build_model()
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(opt["summary_path"] + self._name, self.sess.graph)
        np.random.seed(seed=hash(self._name) % 256)

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

    @abs
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

    def start(self, dir_path):
        """
        Start: new/load a session
        """
        self.sess = tf.Session()
        if os.path.exists(dir_path):
            self.saver.restore(self.sess, dir_path + self._name)
            print("load successfully")
        else:
            self.sess.run(self.init_op)

    def save(self, dir_path):
        """
        Save tensorflow model
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = self.saver.save(self.sess, dir_path + self._name)
        print("Model saved in path: %s" % save_path)

    def save_transition(self, state, action, reward, state_next):
        """
        Save transition to buffer
        """
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_count % self.opt["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_count += 1


class DqnAgent(BaseAgent):
    def __init__(self, name, opt, learning_mode=True):
        super().__init__(name, opt, learning_mode)

    def _build_model(self):
        # w, b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net_' + self._name):
            with tf.variable_scope('phi_net'):
                phi_state_layer_1 = tf.layers.dense(self._state,
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
            tf.summary.histogram('q_values_predict', self.q_values_predict)

            with tf.variable_scope('q_predict'):
                # size of q_value_predict is [BATCH_SIZE, 1]
                action_indices = tf.stack([tf.range(tf.shape(self._action)[0], dtype=tf.int32), self._action], axis=1)
                self.q_value_predict = tf.gather_nd(self.q_values_predict, action_indices)

        with tf.variable_scope('target_net_' + self._name):
            with tf.variable_scope('phi_net'):
                phi_state_next_layer_1 = tf.layers.dense(self._next_state,
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
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

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

        _, cost, summaries = self.sess.run(
            [self.train_op, self.loss, self.merged],
            feed_dict={
                self._state: batch_s,
                self._action: batch_a,
                self._reward: batch_r,
                self._next_state: batch_s_n
            }
        )

        self.epsilon -= self.opt["epsilon_decay"]
        self.writer.add_summary(summaries, global_step)

        if not global_step % 100:
            self.update_q()


"""
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
"""

if __name__ == "__main__":
    # formed data
    a = DqnAgent("DQN_test")
    ac = a.choose_action([[0, 0, 0, 0]])
    print(ac)
