import os
import random
from collections import deque

import numpy as np
import tensorflow as tf

import constants as const
from agents import q_network, drq_network
from sklearn.preprocessing import Normalizer


class DqnAgent(object):
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
        self._target_q = q_network.TargetQNetwork(scope=const.TARGET_Q_SCOPE + name,
                                                  inputs=(self._input_x, self._input_y))
        self._online_q = q_network.OnlineQNetwork(scope=const.ONLINE_Q_SCOPE + name,
                                                  inputs=(self._input_x, self._input_y))

        self._target_q.define_graph()
        self._online_q.define_graph()

        target_q_weights = self._target_q.fullconn_weight + self._target_q.fullconn_bias
        online_q_weights = self._online_q.fullconn_weight + self._online_q.fullconn_bias
        self.replace_target_op = [tf.assign(t, o) for t, o in zip(target_q_weights, online_q_weights)]

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
                                 feed_dict={self._input_x: state})

        if not self._learning_mode or random.random() >= self.epsilon:
            return np.argmax(q_values)
        else:
            # decrease the probability of taking "self-fish" action
            if random.random() >= 0.4:
                return 1
            else:
                return 0
                # return np.random.randint(const.ACTION_SPACE)

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
                    self._input_x: state,  # fixed params
                })

            q_next = self.sess.run(
                self._target_q.outputs[const.MAX_Q_OUTPUT],
                feed_dict={
                    self._input_x: next_state,  # newest params
                })
            target = reward

            if not done:
                target = (reward + self.gamma * q_next)

            target_f = q_eval
            target_f[0][action] = target
            targets[index] = np.reshape(target_f, const.ACTION_SPACE)

        _, loss = self.sess.run([self._online_q.outputs[const.ADAM_OPTIMIZER],
                                 self._online_q.outputs[const.REDUCE_MEAN_LOSS]],
                                feed_dict={self._input_x: states, self._input_y: targets})

        self.epsilon -= const.EPSILON_DECAY

    def update_target_q(self):
        self.sess.run(self.replace_target_op)

    def store_experience(self, state, action, reward, next_state, done):
        transition = [state, action, np.asarray(reward), next_state, done]
        self.memory.append(transition)

    def load(self, name):
        pass

    def save_model(self):
        if not os.path.exists(const.MODEL_SAVE_PATH):
            os.makedirs(const.MODEL_SAVE_PATH)
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)

    def __exit__(self):
        self.sess.close()


class DrqnAgent(object):
    def __init__(self, name, epsilon=const.EPSILON_INIT, learning_rate=0.01, learning_mode=True):
        self._name = name
        self.epsilon = epsilon
        self.gamma = const.GAMMA
        self._learning_rate = learning_rate
        self._learning_mode = learning_mode
        self.memory = deque(maxlen=const.MEMORY_SIZE)

        self._input_x = tf.placeholder(tf.float32,
                                       shape=[None, None, const.STATE_SPACE])
        self._input_y = tf.placeholder(tf.float32,
                                       shape=[None, const.ACTION_SPACE])
        self._dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')

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
            return np.argmax(q_values)
        else:
            # decrease the probability of taking "self-fish" action
            if random.random() >= 0.4:
                return 1
            else:
                return 0
                # return np.random.randint(const.ACTION_SPACE)

    def learn(self):
        # sample batch memory from all memory
        mini_batch = random.sample(self.memory, const.DRQN_MINI_BATCH_SIZE)
        states = np.zeros((const.MINI_BATCH_SIZE, const.STATE_SPACE))
        targets = np.zeros((const.MINI_BATCH_SIZE, const.ACTION_SPACE))

        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            q_eval = self.sess.run(
                self._online_q.outputs[const.Q_VALUE_OUTPUT],
                feed_dict={
                    self._input_x: state,  # fixed params
                })

            q_next = self.sess.run(
                self._target_q.outputs[const.MAX_Q_OUTPUT],
                feed_dict={
                    self._input_x: next_state,  # newest params
                })
            target = reward

            if not done:
                target = (reward + self.gamma * q_next)

            target_f = q_eval
            target_f[0][action] = target

            states[index] = np.reshape(state, const.STATE_SPACE)
            targets[index] = np.reshape(target_f, const.ACTION_SPACE)

        _, loss = self.sess.run([self._online_q.outputs[const.ADAM_OPTIMIZER],
                                 self._online_q.outputs[const.REDUCE_MEAN_LOSS]],
                                feed_dict={self._input_x: states, self._input_y: targets})

        self.epsilon -= const.EPSILON_DECAY

    def update_target_q(self):
        self.sess.run(self.replace_target_op)

    def store_experience(self, state, actions, rewards, terminate):
        transition = (state, actions, rewards, terminate)
        self.memory.append(transition)

    def load(self, name):
        pass

    def save_model(self):
        if not os.path.exists(const.MODEL_SAVE_PATH):
            os.makedirs(const.MODEL_SAVE_PATH)
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)

    def __exit__(self):
        self.sess.close()
