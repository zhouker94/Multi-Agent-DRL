# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 下午8:47
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : constants.py
# @Software: PyCharm Community Edition

ENV_NAME = 'CartPole-v0'

MAX_STEP = 500
MEMORY_SIZE = 5000

ACTION_SPACE = 0
STATE_SPACE = 0

COPY_STEP = 5

TRAINING_EPISODES = 5000
MINI_BATCH_SIZE = 64

TARGET_Q_SCOPE = 'target_q'
ONLINE_Q_SCOPE = 'online_q'
TARGET_Q_COLLECTION = 'target_q_collection'
ONLINE_Q_COLLECTION = 'online_q_collection'

Q_NETWORK_FULLCONN_NUM = 3
Q_NETWORK_WEIGHT_NAME = 'fullconn_weight_'
Q_NETWORK_BIAS_NAME = 'fullconn_bias_'
TRANSITION_QUQUE_ENQUEUE_NAME = 'transition_queue_enqueue'
TRANSITION_QUQUE_DEQUEUE_NAME = 'transition_queue_dequeue'
TRANSITION_QUQUE_SIZE_NAME = 'transition_queue_size'

Q_NETWORK_WEIGHT_SHAPE = [
    [STATE_SPACE, 5],
    [5, 5],
    [5, ACTION_SPACE]
]

Q_NETWORK_BIAS_SHAPE = [
    [1, 5],
    [1, 5],
    [1, ACTION_SPACE]
]

FULLCONN_OUTPUT = 'fullconn_output'
FULLCONN_OUTPUT_WITH_TANH = 'fullconn_output_with_tanh'
FULLCONN_OUTPUT_WITH_SOFTMAX = 'fullconn_output_with_softmax'

Q_VALUE_OUTPUT = 'q_value_output'
ACTION_OUTPUT = 'action_output'
MAX_Q_OUTPUT = 'max_q_output'

REDUCE_MEAN_LOSS = 'reduce_mean_loss'
CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
ADAM_OPTIMIZER = 'adam_optimizer'

EPSILON_INIT = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999

GAMMA = 0.9

LOG_PATH = 'log/'
MODEL_SAVE_PATH = LOG_PATH + 'agent_model/'


def initialize(state_space, action_space):

    global STATE_SPACE, ACTION_SPACE
    STATE_SPACE, ACTION_SPACE = state_space, action_space

    global Q_NETWORK_WEIGHT_SHAPE, Q_NETWORK_BIAS_SHAPE
    Q_NETWORK_WEIGHT_SHAPE = [
        [STATE_SPACE, 5],
        [5, 5],
        [5, ACTION_SPACE]
    ]
    Q_NETWORK_BIAS_SHAPE = [
        5,
        5,
        ACTION_SPACE
    ]
