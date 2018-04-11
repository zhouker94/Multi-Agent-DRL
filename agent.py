import os
import random

import numpy as np
import tensorflow as tf
from collections import deque
import constants as const
import q_network


class DqnAgent(object):
    def __init__(self, name, learning_rate=0.01):
        self._name = name

        self.epsilon = const.EPSILON_INIT
        self.gamma = const.GAMMA
        self._learning_rate = learning_rate

        self.memory = deque(maxlen=2000)

        self._input_x = tf.placeholder(tf.float32,
                                       shape=[None, const.STATE_SPACE])
        self._input_y = tf.placeholder(tf.float32,
                                       shape=[None, const.ACTION_SPACE])
        self._target_q = q_network.TargetQNetwork(scope=const.TARGET_Q_SCOPE + name,
                                                  inputs=(self._input_x, self._input_y))
        self._online_q = q_network.OnlineQNetwork(scope=const.ONLINE_Q_SCOPE + name,
                                                  inputs=(self._input_x, self._input_y))

        target_q_weights = tf.get_collection(const.TARGET_Q_COLLECTION)
        online_q_weights = tf.get_collection(const.ONLINE_Q_COLLECTION)
        self.replace_target_op = [tf.assign(t, o) for t, o in zip(target_q_weights, online_q_weights)]

        self._target_q.define_graph()
        self._online_q.define_graph()

        # The rl_brain should hold a tf session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        # train_writer = tf.summary.FileWriter(const.LOG_PATH, sess.graph)
        if os.path.exists(const.MODEL_SAVE_PATH + self._name):
            print("load successfully")
            self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)

        np.random.seed(seed=int(name))
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def choose_action(self, state):
        if not self.epsilon <= const.EPSILON_MIN:
            self.epsilon *= const.EPSILON_DECAY

        q_values = self.sess.run(self._online_q.outputs[const.Q_VALUE_OUTPUT],
                                 feed_dict={self._input_x: state})

        if random.random() >= self.epsilon:
            return np.argmax(q_values)
        else:
            return np.random.randint(const.ACTION_SPACE)

    def learn(self):
        # sample batch memory from all memory
        mini_batch = random.sample(self.memory, const.MINI_BATCH_SIZE)
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
        if self.epsilon > const.EPSILON_MIN:
            self.epsilon *= const.EPSILON_DECAY

    def update_target_q(self):
        weights = zip(self._online_q.fullconn_weight, self._target_q.fullconn_weight)
        for weight in weights:
            self.sess.run(tf.assign(*weight))
        biases = zip(self._online_q.fullconn_bias, self._target_q.fullconn_bias)
        for bias in biases:
            self.sess.run(tf.assign(*bias))

    def store_experience(self, state, action, reward, next_state, done):
        transition = (state, action, np.asarray(reward), next_state, done)
        self.memory.append(transition)

    def load(self, name):
        pass

    def save_model(self):
        if not os.path.exists(const.MODEL_SAVE_PATH + self._name):
            os.makedirs(const.MODEL_SAVE_PATH + self._name)
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)

    def __exit__(self):
        self.save_model()
        self.sess.close()
