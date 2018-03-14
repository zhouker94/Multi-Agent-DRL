import tensorflow as tf
import constants as const
from utils import Utils
import tf_sub_graph as tsg


class OnlineQNetwork(tsg.TFSubGraph):
    def __init__(self, scope, inputs, learning_rate=0.01):
        super(OnlineQNetwork, self).__init__(scope, inputs)
        self.learning_rate = learning_rate

    def create_variables(self):
        c_names = [const.ONLINE_Q_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES]
        self.fullconn_weight = []
        self.fullconn_bias = []
        for i in range(const.Q_NETWORK_FULLCONN_NUM):
            self.fullconn_weight.append(
                Utils.create_random_normal_variable(name=const.Q_NETWORK_WEIGHT_NAME + str(i),
                                                    scope=self.scope,
                                                    shape=const.Q_NETWORK_WEIGHT_SHAPE[i]))
            self.fullconn_bias.append(
                Utils.create_random_normal_variable(name=const.Q_NETWORK_BIAS_NAME + str(i),
                                                    scope=self.scope,
                                                    shape=const.Q_NETWORK_BIAS_SHAPE[i]))

    def implement_graph(self):
        curr_inputs = self.inputs[0]

        for i in range(const.Q_NETWORK_FULLCONN_NUM):
            self.outputs[const.FULLCONN_OUTPUT + str(i)] = \
                tf.matmul(curr_inputs, self.fullconn_weight[i]) + self.fullconn_bias[i]
            # if is not output layer
            if i < const.Q_NETWORK_FULLCONN_NUM - 1:
                self.outputs[const.FULLCONN_OUTPUT_WITH_TANH + str(i)] = \
                    tf.tanh(self.outputs[const.FULLCONN_OUTPUT + str(i)])
                curr_inputs = self.outputs[const.FULLCONN_OUTPUT_WITH_TANH + str(i)]
            else:
                curr_inputs = self.outputs[const.FULLCONN_OUTPUT + str(i)]

        self.outputs[const.Q_VALUE_OUTPUT] = curr_inputs

        self.outputs[const.REDUCE_MEAN_LOSS] = tf.reduce_mean(tf.squared_difference(self.inputs[1],
                                                                                    self.outputs[const.Q_VALUE_OUTPUT]))

        self.outputs[const.ADAM_OPTIMIZER] = \
            tf.train.AdamOptimizer(1e-4).minimize(self.outputs[const.REDUCE_MEAN_LOSS])


class TargetQNetwork(tsg.TFSubGraph):
    def __init__(self, scope, inputs):
        super(TargetQNetwork, self).__init__(scope, inputs)

    def create_variables(self):
        c_names = [const.TARGET_Q_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES]
        self.fullconn_weight = []
        self.fullconn_bias = []
        for i in range(const.Q_NETWORK_FULLCONN_NUM):
            self.fullconn_weight.append(
                Utils.create_random_normal_variable(name=const.Q_NETWORK_WEIGHT_NAME + str(i),
                                                    scope=self.scope,
                                                    shape=const.Q_NETWORK_WEIGHT_SHAPE[i]))
            self.fullconn_bias.append(
                Utils.create_random_normal_variable(name=const.Q_NETWORK_BIAS_NAME + str(i),
                                                    scope=self.scope,
                                                    shape=const.Q_NETWORK_BIAS_SHAPE[i]))

    def implement_graph(self):
        curr_inputs = self.inputs[0]

        for i in range(const.Q_NETWORK_FULLCONN_NUM):
            self.outputs[const.FULLCONN_OUTPUT + str(i)] = \
                tf.matmul(curr_inputs, self.fullconn_weight[i]) + self.fullconn_bias[i]
            # if is not output layer
            if i < const.Q_NETWORK_FULLCONN_NUM - 1:
                self.outputs[const.FULLCONN_OUTPUT_WITH_TANH + str(i)] = \
                    tf.tanh(self.outputs[const.FULLCONN_OUTPUT + str(i)])
                curr_inputs = self.outputs[const.FULLCONN_OUTPUT_WITH_TANH + str(i)]
            else:
                curr_inputs = self.outputs[const.FULLCONN_OUTPUT + str(i)]

        self.outputs[const.Q_VALUE_OUTPUT] = curr_inputs
