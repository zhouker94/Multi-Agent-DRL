# -*- coding: utf-8 -*-
# @Time    : 2018/7/3 下午8:54
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : rdpg_agent.py
# @Software: PyCharm Community Edition

import argparse
from agents import base_agent
import tensorflow as tf
import numpy as np
import json
import environment
import matplotlib.pyplot as plt


class RDPGAgent(base_agent.BaseAgent):
    def __init__(self, name, opt, learning_mode=True):
        super().__init__(name, opt, learning_mode)
        self.s_buffer = np.zeros((self.opt["memory_size"], 300, self.opt["state_space"]))
        self.r_buffer = np.zeros((self.opt["memory_size"], 1))
        self.a_buffer = np.zeros((self.opt["memory_size"], self.opt["action_space"]))

        self.buffer_item_count = 0

    def _build_model(self):
        self._state = tf.placeholder(tf.float32,
                                     shape=[None, None, self.opt["state_space"]],
                                     name='input_state')
        self._next_state = tf.placeholder(tf.float32,
                                          shape=[None, None, self.opt["state_space"]],
                                          name='input_next_state')

        self._reward = tf.placeholder(tf.float32, [None, 1], name='input_reward')
        self.tau = tf.constant(self.opt["tau"])
        self.a_predict = self.__build_actor_nn(self._state, "predict/actor" + self._name, trainable=True)
        self.a_next = self.__build_actor_nn(self._next_state, "target/actor" + self._name, trainable=False)
        self.q_predict = self.__build_critic(self._state, self.a_predict, "predict/critic" + self._name, trainable=True)
        self.q_next = self.__build_critic(self._next_state, self.a_next, "target/critic" + self._name, trainable=False)
        self.params = []

        for scope in ['predict/actor' + self._name,
                      'target/actor' + self._name,
                      'predict/critic' + self._name,
                      'target/critic' + self._name]:
            self.params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        self.actor_loss = -tf.reduce_mean(self.q_predict)
        self.actor_train_op = tf.train.AdamOptimizer(self.opt["learning_rate"]).minimize(self.actor_loss,
                                                                                         var_list=self.params[0])

        self.q_target = self._reward + self.gamma * self.q_next
        self.critic_loss = tf.losses.mean_squared_error(self.q_target, self.q_predict)
        self.critic_train_op = tf.train.AdamOptimizer(self.opt["learning_rate"] * 2).minimize(self.critic_loss,
                                                                                              var_list=self.params[2])

        self.update_actor = [tf.assign(t_a, (1 - self.tau) * t_a + self.tau * p_a) for p_a, t_a in zip(self.params[0],
                                                                                                       self.params[1])]

        self.update_critic = [tf.assign(t_c, (1 - self.tau) * t_c + self.tau * p_c) for p_c, t_c in zip(self.params[2],
                                                                                                        self.params[3])]

    def __build_actor_nn(self, state, scope, trainable=True):
        w_init, b_init = tf.random_normal_initializer(.0, .001), tf.constant_initializer(.001)

        with tf.variable_scope(scope):
            batch_norm_state = tf.layers.batch_normalization(state)

            e_gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                 for i in range(len(self.opt["gru_nodes_nums"]))])
            e_gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                 for i in range(len(self.opt["gru_nodes_nums"]))])

            _, h_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=e_gru_cell_fw,
                                                         cell_bw=e_gru_cell_bw,
                                                         inputs=batch_norm_state,
                                                         dtype=tf.float32)

            rnn_output = tf.concat([h[-1] for h in h_state[0]], axis=1)

            phi_state = tf.layers.dense(rnn_output,
                                        32,
                                        tf.nn.relu,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        trainable=trainable)

            action_prob = tf.layers.dense(phi_state,
                                          self.opt["action_space"],
                                          tf.nn.sigmoid,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init,
                                          trainable=trainable)

            return tf.multiply(action_prob, self.opt["action_upper_bound"])

    def __build_critic(self, state, action, scope, trainable=True):
        w_init, b_init = tf.random_normal_initializer(.0, .001), tf.constant_initializer(.001)

        with tf.variable_scope(scope):
            batch_norm_state = tf.layers.batch_normalization(state)

            e_gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                 for i in range(len(self.opt["gru_nodes_nums"]))])
            e_gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                 for i in range(len(self.opt["gru_nodes_nums"]))])

            _, phi_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=e_gru_cell_fw,
                                                           cell_bw=e_gru_cell_bw,
                                                           inputs=batch_norm_state,
                                                           dtype=tf.float32)

            phi_action = tf.layers.dense(action,
                                         32,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init,
                                         trainable=trainable)

            q_value = tf.layers.dense(tf.nn.relu(phi_state + phi_action),
                                      1,
                                      kernel_initializer=w_init,
                                      bias_initializer=b_init,
                                      trainable=trainable)

            return q_value

    def save_transition(self, states, actions, rewards):
        # transition = np.hstack((state, action, [reward], state_next))
        index = self.buffer_item_count % self.opt["memory_size"]
        self.s_buffer[index, :] = states
        self.r_buffer[index, :] = rewards
        self.a_buffer[index, :] = actions

        self.buffer_item_count += 1

    def get_sample_batch(self):
        indices = np.random.choice(self.opt["memory_size"], size=self.opt["batch_size"])
        state = self.s_buffer[indices, :, :]
        action = self.a_buffer[indices, :]
        reward = self.r_buffer[indices, :]
        indices = np.add(indices, np.ones_like(indices))
        state_next = self.s_buffer[indices, :, :]
        return state, action, reward, state_next

    def learn(self, global_step):
        self.sess.run([self.update_actor, self.update_critic])

        state, action, reward, state_next = self.get_sample_batch()

        self.sess.run(self.actor_train_op, {
            self._state: state})

        self.sess.run(self.critic_train_op, {
            self._state: state, self.a_predict: action, self._reward: reward, self._next_state: state_next
        })

        self.epsilon -= self.opt["epsilon_decay"]

    def choose_action(self, state, action_upper_bound):
        action = self.sess.run(self.a_predict, {self._state: state})
        exploration_scale = 100 * self.epsilon
        action = np.clip(np.random.normal(action[0], exploration_scale), 0, action_upper_bound)
        return action[0]


'''
def agent_test():
    conf = json.load(open('../config.json', 'r'))
    opt = conf["rdpg"]
    player = RDPGAgent("RDPG_", opt)
    player.start()
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=1)
    parser.add_argument('--sustainable_weight', type=float, default=0.5)
    parser.add_argument('--is_test', type=bool, default=False)
    parsed_args = parser.parse_args()

    conf = json.load(open('../config.json', 'r'))
    training_conf = conf["training_config"]

    env_conf = conf["env_config"]
    env_conf["sustain_weight"] = parsed_args.sustainable_weight
    training_conf["num_agents"] = parsed_args.n_agents
    env = environment.GameEnv(env_conf)

    dir_conf, option = conf["dir_config"], conf["rdpg"]
    dir_conf["model_save_path"] = dir_conf["model_save_path"] + '_' + \
                                  str(env_conf["sustain_weight"]) + '_' + \
                                  str(training_conf["num_agents"]) + '/'

    avg_scores = []
    global_step = 0

    # -------------- start train mode --------------

    if not parsed_args.is_test:

        agent_list = []
        for i in range(training_conf["num_agents"]):
            player = RDPGAgent("RDPG_" + str(i), option)
            player.start(dir_path=dir_conf["model_save_path"])
            agent_list.append(player)

        for epoch in range(training_conf["train_epochs"]):
            if agent_list[0].epsilon <= option["min_epsilon"]:
                break

            # state -> [X, Pi]
            state = env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf["num_agents"]
            score = 0

            states = np.zeros((training_conf["max_round"], option['action_space']))
            rewards = [[] for _ in range(training_conf["num_agents"])]
            actions = [[] for _ in range(training_conf["num_agents"])]

            for time in range(training_conf["max_round"]):

                for index, player in enumerate(agent_list):
                    efforts[index] = player.choose_action(np.expand_dims(state, axis=0),
                                                          env.common_resource_pool / training_conf["num_agents"])

                next_state, rewards, done = env.step(efforts)

                states[time, :] = state

                score += sum(rewards)

                state = next_state

                global_step += 1

                if done:
                    break

            [player.save_transition(states, actions[index], rewards[index])
             for index, player in enumerate(agent_list)]

            if not epoch % 2:
                [player.learn(global_step) for player in agent_list]

            score /= training_conf["num_agents"]

            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(epoch, training_conf["train_epochs"], score, agent_list[0].epsilon))

            avg_scores.append(score)

        for a in agent_list:
            a.save(dir_path=dir_conf["model_save_path"])
            a.sess.close()

        # -------------- save results --------------
        plt.switch_backend('agg')
        plt.plot(avg_scores)
        plt.interactive(False)
        plt.xlabel('Epoch')
        plt.ylabel('Avg score')
        plt.savefig(dir_conf["model_save_path"] + 'rdpg_training_plot')

        with open(dir_conf["model_save_path"] + 'rdpg_avg_score.txt', "w+") as f:
            for r in avg_scores:
                f.write(str(r) + '\n')
