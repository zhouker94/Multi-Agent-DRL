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
                                        dtypes=[tf.float32, tf.float32],
                                        shapes=[(const.STATE_SPACE), ()])
        self.sess = sess
        
        self.state = tf.placeholder(dtype=tf.float32, 
                                    shape=(const.STATE_SPACE))
        self.target = tf.placeholder(dtype=tf.float32,
                                     shape=())

        self.tq_enqueue_op = self.tq.enqueue([self.state, self.target],
                                             name=const.TRANSITION_QUQUE_ENQUEUE_NAME)
        self.tq_dequeue_op = self.tq.dequeue_many(const.MINI_BATCH_SIZE,
                                                  name=const.TRANSITION_QUQUE_DEQUEUE_NAME)

        self.tq_get_size_op = self.tq.size(name=const.TRANSITION_QUQUE_SIZE_NAME)

    def enqueue(self, val):
        self.sess.run(self.tq_enqueue_op, feed_dict={self.state: val[0],
                                                     self.target: val[1]})

    def dequeue(self):
        return self.sess.run(self.tq_dequeue_op)

    def get_size(self):
        return self.sess.run(self.tq_get_size_op)

