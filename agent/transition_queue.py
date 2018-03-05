# -*- coding: utf-8 -*-
# @Time    : 2018/2/18 下午10:28
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : transition_queue.py
# @Software: PyCharm Community Edition

import tensorflow as tf
import constants as const


class TransitionQueue(object):
    def __init__(self, sess):
        self.length = 0
        self.tq = tf.RandomShuffleQueue(const.REPLAY_BUFFER_LENGTH,
                                        const.MIN_TRANSITIONS_IN_BUFFER,
                                        dtypes=[tf.float32, tf.int32, tf.float32, tf.float32, tf.bool],
                                        shapes=[(1, const.STATE_SPACE), (), (), (1, const.STATE_SPACE), ()])
        self.sess = sess

        self.state = tf.placeholder(tf.float32, shape=(1, const.STATE_SPACE))
        self.action = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)
        self.next_state = tf.placeholder(tf.float32, shape=(1, const.STATE_SPACE))
        self.done = tf.placeholder(tf.bool)

        self.tq_enqueue_op = self.tq.enqueue([self.state, self.action, self.reward, self.next_state, self.done],
                                             name=const.TRANSITION_QUQUE_ENQUEUE_NAME)

        self.tq_dequeue_op = self.tq.dequeue_many(const.MINI_BATCH_SIZE,
                                                  name=const.TRANSITION_QUQUE_DEQUEUE_NAME)

        self.tq_get_size_op = self.tq.size(name=const.TRANSITION_QUQUE_SIZE_NAME)

    def enqueue(self, val):
        self.sess.run(self.tq_enqueue_op, feed_dict={self.state: val[0],
                                                     self.action: val[1],
                                                     self.reward: val[2],
                                                     self.next_state: val[3],
                                                     self.done: val[4]})

    def dequeue(self):
        return self.sess.run(self.tq_dequeue_op)

    def get_size(self):
        return self.sess.run(self.tq_get_size_op)
