import os
import random

import tensorflow as tf
import constants as const
import q_network
import transition_queue as tq


class DqnAgent(object):
    def __init__(self, name, learning=False):
        self._name = name

        self.epsilon = const.EPSILON_INIT
        self.learning = learning

        self._input_x = tf.placeholder(tf.float32,
                                       [None, const.STATE_SPACE])
        self._input_y = tf.placeholder(tf.float32,
                                       [None, const.ACTION_SPACE])
        self.target_q = q_network.QNetwork(scope=const.TARGET_Q_SCOPE,
                                           inputs=(self._input_x, self._input_y))
        self.online_q = q_network.QNetwork(scope=const.ONLINE_Q_SCOPE,
                                           inputs=(self._input_x, self._input_y))

        self.target_q.define_graph()
        self.online_q.define_graph()

        # The agent should hold a tf session
        self.sess = tf.Session()
        self.replay_buffer = tq.TransitionQueue(self.sess)

        self.saver = tf.train.Saver()
        # train_writer = tf.summary.FileWriter(const.LOG_PATH, sess.graph)
        if os.path.exists(const.MODEL_SAVE_PATH + self._name):
            self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def choose_action(self, state):
        if not self.epsilon <= const.EPSILON_MIN:
            self.epsilon *= const.EPSILON_DECAY

        q_values = self.sess.run(self.online_q.outputs[const.Q_VALUE_OUTPUT],
                                 feed_dict={self._input_x: state})
        if random.random() <= self.epsilon:
            return random.randint(0, const.ACTION_SPACE - 1)
        else:
            actions = self.sess.run(tf.argmax(q_values))
            return random.choice(actions)

    def update_online_q(self):
        transitions = list(self.replay_buffer.dequeue())
        print(transitions)
        # TODO: training Step
        # self.sess.run(self.online_q.outputs[const.ADAM_OPTIMIZER], feed_dict=)
        pass

    def update_target_q(self):
        pass

    def store_experience(self, state, action, reward, next_state, done):
        print(state.dtype, action, reward, next_state.dtype, done)
        self.replay_buffer.enqueue((state, action, reward, next_state, done))

    def load(self, name):
        pass

    def save_model(self):
        save_path = self.saver.save(self.sess, const.MODEL_SAVE_PATH + self._name)
        print("Model saved in path: %s" % save_path)

    def __exit__(self):
        self.save_model()
        self.sess.close()