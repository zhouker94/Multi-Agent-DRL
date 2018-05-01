# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 下午8:47
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : env_constants.py
# @Software: PyCharm Community Edition

ENV_NAME = 'CartPole-v0'

MAX_STEP = 100
MEMORY_SIZE = 500

TEST_STEP = 1

ACTION_SPACE = 0
STATE_SPACE = 0

COPY_STEP = 20

TRAINING_EPISODES = 1000
MINI_BATCH_SIZE = 128
DRQN_MINI_BATCH_SIZE = 5

TARGET_Q_SCOPE = 'target_q'
ONLINE_Q_SCOPE = 'online_q'

Q_NETWORK_FULLCONN_NUM = 4
Q_NETWORK_WEIGHT_NAME = 'fullconn_weight_'
Q_NETWORK_BIAS_NAME = 'fullconn_bias_'

Q_NETWORK_WEIGHT_SHAPE = None
Q_NETWORK_BIAS_SHAPE = None

FULLCONN_OUTPUT = 'fullconn_output'
FULLCONN_OUTPUT_WITH_TANH = 'fullconn_output_with_tanh'
FULLCONN_OUTPUT_WITH_SOFTMAX = 'fullconn_output_with_softmax'

# GRU config
DRQ_NETWORK_GRU_LAYER_NUM = 3
DRQ_NETWORK_UNITS_NUMBER = [128, 32, 2]
ONLINE_DRQ_NETWORK_WEIGHT_NAME = "online_drqn_weight_"
TARGET_DRQ_NETWORK_WEIGHT_NAME = "target_drqn_weight"

Q_VALUE_OUTPUT = 'q_value_output'
ACTION_OUTPUT = 'action_output'
MAX_Q_OUTPUT = 'max_q_output'

REDUCE_MEAN_LOSS = 'reduce_mean_loss'
CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
ADAM_OPTIMIZER = 'adam_optimizer'

EPSILON_INIT = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.001

GAMMA = 0.9

N_AGENTS = 0
LOG_PATH = 'log/'
MODEL_SAVE_PATH = LOG_PATH + 'agent_model/'


def initialize(state_space, action_space, n_agents, weight):
    global STATE_SPACE, ACTION_SPACE
    STATE_SPACE, ACTION_SPACE = state_space, action_space

    global Q_NETWORK_WEIGHT_SHAPE, Q_NETWORK_BIAS_SHAPE, N_AGENTS, WEIGHT, LOG_PATH, MODEL_SAVE_PATH
    Q_NETWORK_WEIGHT_SHAPE = [
        [STATE_SPACE, 16],
        [16, 32],
        [32, 16],
        [16, ACTION_SPACE]
    ]
    Q_NETWORK_BIAS_SHAPE = [
        16,
        32,
        16,
        ACTION_SPACE
    ]
    N_AGENTS = n_agents
    WEIGHT = weight
    LOG_PATH = "train_log_with_weight_" + str(WEIGHT) + "_agents_" + str(N_AGENTS) + "/"
    MODEL_SAVE_PATH = LOG_PATH + 'agent_model/'
    print("Log path is:", LOG_PATH)
    print("The number of agents is:", N_AGENTS)
    print("The sustainable weight is:", WEIGHT)

    global INIT_EFFORT, MIN_EFFORT, MAX_EFFORT
    INIT_EFFORT = 300 / N_AGENTS
    MIN_EFFORT = 100 / N_AGENTS
    MAX_EFFORT = 500 / N_AGENTS


# environment configuration
RESOURCE_CAPACITY_N_MAX = 1000.0
RESOURCE_CAPACITY_INIT = 1000.0

REPLENISHMENT_RATE = 0.5

ALPHA = 0.35
BETA = 0.4
COST_C = 0.5

# important parameter in this project
WEIGHT = 0.5

MIN_INCREMENT = 10

INIT_EFFORT = 0
MIN_EFFORT = 0
MAX_EFFORT = 0
