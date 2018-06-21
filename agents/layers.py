# -*- coding: utf-8 -*-
# @Time    : 2018/6/21 下午9:49
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : layers.py
# @Software: PyCharm Community Edition


import tensorflow as tf


class Layer:
    @staticmethod
    def coattention(encode_c, encode_q):
        # (batch_size, hidden_size，question)
        variation_q = tf.transpose(encode_q, [0, 2, 1])
        # [batch, c length, q length]w
        L = tf.matmul(encode_c, variation_q)
        L_t = tf.transpose(L, [0, 2, 1])
        # normalize with respect to question
        a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
        # normalize with respect to context
        a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
        # summaries with respect to question, (batch_size, question, hidden_size)
        c_q = tf.matmul(a_q, encode_c)
        c_q_emb = tf.concat((variation_q, tf.transpose(c_q, [0, 2, 1])), 1)
        # summaries of previous attention with respect to context
        c_d = tf.matmul(c_q_emb, a_c, adjoint_b=True)
        # coattention context [batch_size, context+1, 3*hidden_size]
        co_att = tf.concat((encode_c, tf.transpose(c_d, [0, 2, 1])), 2)
        return co_att

    def self_attention(encoder_output):
        """
        # (batch, words, encode) using bi-linear
        W_1 = tf.layers.dense(encoder_output, 128, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001), activation=tf.nn.relu)
        W_2 = tf.layers.dense(encoder_output, 128, use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001), activation=tf.nn.relu)
        # (batch, word, word)
        W_1_2 = tf.matmul(W_1, tf.transpose(W_2, [0, 2, 1]))
        """
        norm_layer = tf.contrib.layers.layer_norm(encoder_output)
        W_1_2 = tf.matmul(norm_layer, tf.transpose(norm_layer, [0, 2, 1]))
        return tf.matmul(tf.nn.softmax(W_1_2), norm_layer)