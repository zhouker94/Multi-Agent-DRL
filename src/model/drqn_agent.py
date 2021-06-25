#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:18
# @Author  : Hanwei Zhu
# @File    : drqn_agent.py

import argparse
import random

import numpy as np
import sys
import tensorflow as tf

import base_agent
import json

sys.path.append("../")
import environment


class DRQNAgent(base_agent.BaseAgent):
    def __init__(self, name, opt, learning_mode=True):
        super().__init__(name, opt, learning_mode)
        self.buffer = np.zeros((self.opt["memory_size"], self.opt["max_round"],
                                self.opt["state_space"] + 1 + 1))

        self.buffer_count = 0

    def _build_model(self):
        self._state = tf.placeholder(tf.float32,
                                     shape=[None, None, self.opt["state_space"]],
                                     name='input_state')
        self._next_state = tf.placeholder(tf.float32,
                                          shape=[None, None, self.opt["state_space"]],
                                          name='input_next_state')

        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net_' + self._name):
            batch_norm_state = tf.layers.batch_normalization(self._state)

            with tf.variable_scope('phi_net'):
                e_gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                     for i in range(len(self.opt["gru_nodes_nums"]))])

                e_gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                     for i in range(len(self.opt["gru_nodes_nums"]))])

                _, (fw_h_state, bw_h_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=e_gru_cell_fw,
                                                                              cell_bw=e_gru_cell_bw,
                                                                              inputs=batch_norm_state,
                                                                              dtype=tf.float32)

                phi_state_output = tf.concat([fw_h_state[-1], bw_h_state[-1]], axis=1)

            self.q_values_predict = tf.layers.dense(phi_state_output,
                                                    self.opt["action_space"],
                                                    kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer,
                                                    name='Q_predict')

            # tf.summary.histogram('q_values_predict', self.q_values_predict)

            with tf.variable_scope('q_predict'):
                # size of q_value_predict is [BATCH_SIZE, 1]
                action_indices = tf.stack([tf.range(tf.shape(self._action)[0], dtype=tf.int32), self._action], axis=1)
                self.q_value_predict = tf.gather_nd(self.q_values_predict, action_indices)
                self.action_output = tf.argmax(self.q_values_predict)

        with tf.variable_scope('target_net_' + self._name):
            batch_norm_next_state = tf.layers.batch_normalization(self._next_state)

            with tf.variable_scope('phi_net'):
                t_gru_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                     for i in range(len(self.opt["gru_nodes_nums"]))])
                t_gru_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.contrib.rnn.GRUCell(num_units=self.opt["gru_nodes_nums"][i], activation=tf.nn.relu)
                     for i in range(len(self.opt["gru_nodes_nums"]))])
                _, (fw_h_state, bw_h_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=t_gru_cell_fw,
                                                                              cell_bw=t_gru_cell_bw,
                                                                              inputs=batch_norm_next_state,
                                                                              dtype=tf.float32)

                phi_next_state_output = tf.concat([fw_h_state[-1], bw_h_state[-1]], axis=1)

            self.q_values_target = tf.layers.dense(phi_next_state_output,
                                                   self.opt["action_space"],
                                                   kernel_initializer=w_initializer,
                                                   bias_initializer=b_initializer,
                                                   name='Q_target')

            with tf.variable_scope('q_real'):
                # size of q_value_real is [BATCH_SIZE, 1]
                q_value_max = tf.reduce_max(self.q_values_target, axis=1)
                q_value_real = self._reward + self.gamma * q_value_max
                self.q_value_real = tf.stop_gradient(q_value_real)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value_real, self.q_value_predict, name='mse'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss)

        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net_" + self._name)
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net_" + self._name)

        with tf.variable_scope('soft_replacement'):
            self.update_q_net = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

    def update_q(self):
        """
        Copy weights from eval_net to target_net
        """
        self.sess.run(self.update_q_net)

    def learn(self, global_step):
        if self.buffer_count > self.opt["memory_size"]:
            sample_indices = np.random.choice(self.opt["memory_size"], size=self.opt["batch_size"])
        else:
            sample_indices = np.random.choice(self.buffer_count, size=self.opt["batch_size"])

        batch = self.buffer[sample_indices, :, :]
        batch_s = batch[:, :, :self.opt["state_space"]]
        batch_a = batch[:, :, self.opt["state_space"]]
        batch_r = batch[:, :, self.opt["state_space"] + 1]

        for step in range(self.opt["max_round"]):
            _, cost = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self._state: batch_s[:, :(step + 1)],
                    self._action: batch_a[:, step],
                    self._reward: batch_r[:, step],
                    self._next_state: batch_s[:, :(step + 2)],
                }
            )

        self.epsilon -= self.opt["epsilon_decay"]
        # self.writer.add_summary(summaries, global_step)

        if not global_step % 100:
            self.update_q()

    def choose_action(self, state):
        """
        Choose an action
        """
        if not self._learning_mode or random.random() >= self.epsilon:
            return np.argmax(self.sess.run(self.action_output,
                                           feed_dict={self._state: state}))
        else:
            return np.random.randint(self.opt["action_space"])

    def save_transition(self, step, state, action, reward, done):
        """
        Save transition to buffer
        """
        transition = np.hstack((state, [action, reward]))
        index = self.buffer_count % self.opt["memory_size"]
        self.buffer[index, step, :] = transition
        
        if done:
            self.buffer_count += 1

'''
def agent_test():
    conf = json.load(open('../config.json', 'r'))
    opt = conf["drqn"]
    player = DRQNAgent("DRQN_", opt)
    player.start()
'''

if __name__ == "__main__":
    # agent_test()

    # -------------- parameters initialize --------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=1)
    parser.add_argument('--sustainable_weight', type=float, default=0.5)
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--replenishment_rate', type=float, default=0.5)
    parsed_args = parser.parse_args()

    conf = json.load(open('../config/config.json', 'r'))
    training_conf = conf["training_config"]
    training_conf["num_agents"] = parsed_args.n_agents

    env_conf = conf["env_config"]
    env_conf["sustain_weight"] = parsed_args.sustainable_weight
    env_conf["replenishment_rate"] = parsed_args.replenishment_rate
    env = environment.GameEnv(env_conf)

    dir_conf, agent_opt = conf["dir_config"], conf["drqn"]
    dir_conf["model_save_path"] = dir_conf["model_save_path"] + '_' + \
                                  str(env_conf["sustain_weight"]) + '_' + \
                                  str(training_conf["num_agents"]) + '/'

    avg_scores = []
    global_step = 0
    phi_states = [np.zeros((training_conf["max_round"], agent_opt["state_space"])) for _ in
                 range(training_conf["num_agents"])]

    # -------------- start train mode --------------

    if not parsed_args.is_test:

        agent_list = []
        for i in range(training_conf["num_agents"]):
            player = DRQNAgent("DRQN_" + str(i), agent_opt)
            player.start(dir_path=dir_conf["model_save_path"])
            agent_list.append(player)

        for epoch in range(training_conf["train_epochs"]):
            if agent_list[0].epsilon <= agent_opt["min_epsilon"]:
                break

            env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf["num_agents"]

            score = 0

            for time in range(training_conf["max_round"]):
                # actions -> [Increase effort, Decrease effort, IDLE]
                actions = [0] * training_conf["num_agents"]

                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(phi_states[index], axis=0))
                    actions[index] = action

                    # increase
                    if action == 0:
                        efforts[index] += training_conf["min_increment"]
                    # decrease
                    elif action == 1:
                        efforts[index] -= training_conf["min_increment"]

                    if efforts[index] <= 1:
                        efforts[index] = 1

                next_states, rewards, done = env.step(efforts)

                score += sum(rewards)

                for index, player in enumerate(agent_list):
                    phi_states[index][time, :] = np.asarray(next_states[index])
                    player.save_transition(time, next_states[index], actions[index], rewards[index], done)

                global_step += 1

                if done:
                    break

            if not epoch % 5:
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
        plt.savefig(dir_conf["model_save_path"] + 'drqn_training_plot')

        with open(dir_conf["model_save_path"] + 'drqn_avg_score.txt', "w+") as f:
            for r in avg_scores:
                f.write(str(r) + '\n')
