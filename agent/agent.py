import os
import random

import numpy as np
import tensorflow as tf
import constants as const
import q_network


class DqnAgent(object):
    def __init__(self, name, learning_rate=0.01):
        self._name = name

        self.epsilon = const.EPSILON_INIT
        self.gamma = const.GAMMA
        self._learning_rate = learning_rate

        self._memory = []
        self._memory_counter = 0

        self._input_x = tf.placeholder(tf.float32,
                                       shape=[None, const.STATE_SPACE])
        self._input_y = tf.placeholder(tf.float32,
                                       shape=[None, const.ACTION_SPACE])
        self._target_q = q_network.TargetQNetwork(scope=const.TARGET_Q_SCOPE,
                                                  inputs=(self._input_x, self._input_y))
        self._online_q = q_network.OnlineQNetwork(scope=const.ONLINE_Q_SCOPE,
                                                  inputs=(self._input_x, self._input_y))

        target_q_weights = tf.get_collection(const.TARGET_Q_COLLECTION)
        online_q_weights = tf.get_collection(const.ONLINE_Q_COLLECTION)
        self.replace_target_op = [tf.assign(t, o) for t, o in zip(target_q_weights, online_q_weights)]

        self._target_q.define_graph()
        self._online_q.define_graph()

        # The agent should hold a tf session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        # train_writer = tf.summary.FileWriter(const.LOG_PATH, sess.graph)
        if os.path.exists(const.MODEL_SAVE_PATH + self._name):
            self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)

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
            return np.random.randint(0, const.ACTION_SPACE - 1)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # TODO: training Step
        map_state_op = tf.map_fn(lambda x: tf.reshape(x, [const.STATE_SPACE]), states)
        stack_state_op = tf.stack(map_state_op)
        batch_x = self.sess.run(stack_state_op)

        map_targets_op = tf.map_fn(lambda y: tf.reshape(y, [const.ACTION_SPACE]), targets)
        stack_targets_op = tf.stack(map_targets_op)
        batch_y = self.sess.run(stack_targets_op)

        self.sess.run(self._online_q.outputs[const.ADAM_OPTIMIZER],
                      feed_dict={self._input_x: batch_x, self._input_y: batch_y})
        pass

    def update_target_q(self):
        weights = zip(self._online_q.fullconn_weight, self._target_q.fullconn_weight)
        for weight in weights:
            self.sess.run(tf.assign(*weight))
        biases = zip(self._online_q.fullconn_bias, self._target_q.fullconn_bias)
        for bias in biases:
            self.sess.run(tf.assign(*bias))

    def store_experience(self, state, action, reward, next_state, done):
        transition = np.hstack((state, action, reward, next_state, done))
        # replace the old memory with new memory
        self._memory_counter %= const.MEMORY_SIZE
        self._memory[self._memory_counter, :] = transition
        self._memory_counter += 1

    def load(self, name):
        pass

    def save_model(self):
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)

    def __exit__(self):
        self.save_model()
        self.sess.close()
