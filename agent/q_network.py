import tensorflow as tf
import constants as const
from utils import Utils
import tf_sub_graph as tsg


class QNetwork(tsg.TFSubGraph):
    def __init__(self, scope, inputs):
        super(QNetwork, self).__init__(scope, inputs)

    def create_variables(self):
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

        self.outputs[const.Q_VALUE_OUTPUT] = tf.nn.softmax(curr_inputs)
        self.outputs[const.MAX_Q_OUTPUT] = tf.reduce_max(self.outputs[const.Q_VALUE_OUTPUT],
                                                         axis=1)
        self.outputs[const.ACTION_OUTPUT] = \
            tf.argmax(self.outputs[const.Q_VALUE_OUTPUT])
        
        self.outputs[const.CROSS_ENTROPY_LOSS] = \
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.inputs[1],
                logits=curr_inputs)

        self.outputs[const.ADAM_OPTIMIZER] = \
            tf.train.AdamOptimizer(1e-4).minimize(self.outputs[const.CROSS_ENTROPY_LOSS])


    def fit(self, batch_x, batch_y):
        pass

    def predict(self, batch_x):
        pass

