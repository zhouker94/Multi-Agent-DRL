import os
import random

import numpy as np
import tensorflow as tf
import constants as const
import q_network
import transition_queue as tq


class DqnAgent(object):
    def __init__(self, name, learning=False):
        self._name = name

        self.epsilon = const.EPSILON_INIT
        self.gamma = const.GAMMA
        self._learning = learning

        self._input_x = tf.placeholder(tf.float32,
                                       shape=[None, const.STATE_SPACE])
        self._input_y = tf.placeholder(tf.float32,
                                       shape=[None, const.ACTION_SPACE])
        self._target_q = q_network.QNetwork(scope=const.TARGET_Q_SCOPE,
                                           inputs=(self._input_x, self._input_y))
        self._online_q = q_network.QNetwork(scope=const.ONLINE_Q_SCOPE,
                                           inputs=(self._input_x, self._input_y))

        self._target_q.define_graph()
        self._online_q.define_graph()

        # The agent should hold a tf session
        self.sess = tf.Session()
        self.memory = np.zeros((const.MEMORY_SIZE, n_features*2+2))

        self.saver = tf.train.Saver()
        # train_writer = tf.summary.FileWriter(const.LOG_PATH, sess.graph)
        if os.path.exists(const.MODEL_SAVE_PATH + self._name):
            self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def choose_action(self, state):
        if not self.epsilon <= const.EPSILON_MIN:
            self.epsilon *= const.EPSILON_DECAY
        
        actions = self.sess.run(self._online_q.outputs[const.ACTION_OUTPUT],
                                 feed_dict={self._input_x: state})
        
        if random.random() <= self.epsilon:
            return random.randint(0, const.ACTION_SPACE - 1)
        else:
            return random.choice(actions)

    def update_online_q(self):
        transitions = self.replay_buffer.dequeue()
        states = transitions[0]
        targets = transitions[1]
        
        # TODO: training Step
        map_state_op = tf.map_fn(lambda x: tf.reshape(x, [const.STATE_SPACE]), states)
        stack_state_op = tf.stack(map_state_op)
        batch_x = self.sess.run(stack_state_op)

        map_targets_op = tf.map_fn(lambda y: tf.reshape(y, [const.ACTION_SPACE]), targets)
        statck_targets_op = tf.stack(map_targets_op)
        batch_y = self.sess.run(statck_targets_op)
        
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
        targets = self.sess.run(self._online_q.outputs[const.Q_VALUE_OUTPUT],
                                feed_dict={self._input_x: state})
        next_max_q = self.sess.run(self._target_q.outputs[const.MAX_Q_OUTPUT],
                                     feed_dict={self._input_x: next_state})
        targets[0][action] = reward
        if not done:
            targets[0][action] = reward + self.gamma * next_max_q
        
        self.replay_buffer.enqueue([state, targets])

    def load(self, name):
        pass

    def save_model(self):
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)

    def __exit__(self):
        self.save_model()
        self.sess.close()

