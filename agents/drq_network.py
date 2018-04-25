#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 15:30
# @Author  : Hanwei Zhu
# @File    : drq_network.py

from agents import tf_sub_graph as tsg
from agents.utils import Utils
import constants as const
import tensorflow as tf


class OnlineDRQNetwork(tsg.TFSubGraph):
    def __init__(self, scope, inputs, dropout_keep_prob, learning_rate=0.01):
        super(OnlineDRQNetwork, self).__init__(scope, inputs)
        self.rnn_cells = []
        self.rnn_layer = None
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob

    def create_variables(self):
        for i in range(const.DRQ_NETWORK_GRU_LAYER_NUM):
            self.rnn_cells.append(
                Utils.create_gru_cell(name=const.ONLINE_DRQ_NETWORK_WEIGHT_NAME + str(i),
                                      units_number=const.DRQ_NETWORK_UNITS_NUMBER[i]))
        layered_cell = tf.contrib.rnn.MultiRNNCell(self.rnn_cells)
        self.rnn_layer = tf.contrib.rnn.DropoutWrapper(layered_cell,
                                                       input_keep_prob=self.dropout_keep_prob,
                                                       output_keep_prob=self.dropout_keep_prob,
                                                       state_keep_prob=self.dropout_keep_prob,
                                                       )

    def implement_graph(self):
        curr_inputs = self.inputs[0]
        curr_inputs, _ = tf.nn.dynamic_rnn(cell=self.rnn_layer, inputs=curr_inputs, dtype=tf.float32)

        self.outputs[const.Q_VALUE_OUTPUT] = curr_inputs

        self.outputs[const.REDUCE_MEAN_LOSS] = tf.reduce_mean(tf.squared_difference(self.inputs[1],
                                                                                    self.outputs[const.Q_VALUE_OUTPUT]))

        self.outputs[const.ADAM_OPTIMIZER] = \
            tf.train.AdamOptimizer(1e-4).minimize(self.outputs[const.REDUCE_MEAN_LOSS])


class TargetDRQNetwork(tsg.TFSubGraph):
    def __init__(self, scope, inputs, dropout_keep_prob, learning_rate=0.01):
        super(TargetDRQNetwork, self).__init__(scope, inputs)
        self.rnn_cells = []
        self.rnn_layer = None
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob

    def create_variables(self):
        for i in range(const.DRQ_NETWORK_GRU_LAYER_NUM):
            self.rnn_cells.append(
                Utils.create_gru_cell(name=const.TARGET_DRQ_NETWORK_WEIGHT_NAME + str(i),
                                      units_number=const.DRQ_NETWORK_UNITS_NUMBER[i]))
        layered_cell = tf.contrib.rnn.MultiRNNCell(self.rnn_cells)
        self.rnn_layer = tf.contrib.rnn.DropoutWrapper(layered_cell,
                                                       input_keep_prob=self.dropout_keep_prob,
                                                       output_keep_prob=self.dropout_keep_prob,
                                                       state_keep_prob=self.dropout_keep_prob,
                                                       )

    def implement_graph(self):
        curr_inputs = self.inputs[0]
        curr_inputs, _ = tf.nn.dynamic_rnn(cell=self.rnn_layer, inputs=curr_inputs, dtype=tf.float32)

        self.outputs[const.Q_VALUE_OUTPUT] = curr_inputs

        self.outputs[const.REDUCE_MEAN_LOSS] = tf.reduce_mean(tf.squared_difference(self.inputs[1],
                                                                                    self.outputs[const.Q_VALUE_OUTPUT]))

        self.outputs[const.ADAM_OPTIMIZER] = \
            tf.train.AdamOptimizer(1e-4).minimize(self.outputs[const.REDUCE_MEAN_LOSS])
