#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 15:09
# @Author  : Hanwei Zhu
# @File    : dqn.py

import argparse
import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
from model import base_model
import random
import os


class DQNModel(base_model.BaseModel):
    def __init__(self, aid, config):
        super().__init__(aid, config)

        self.buffer = np.zeros(
            (self.config["memory_size"],
             self.config["state_space"] + 1 + 1 + self.config["state_space"])
        )
        self.epsilon = self.config["init_epsilon"]
        self.buffer_count = 0

    def _build_graph(self):
        # w, b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net_' + self.model_id):
            # batch_norm_state = tf.contrib.layers.batch_norm(self._state)
            with tf.variable_scope('phi_net'):
                phi_state_layer_1 = tf.layers.dense(
                    self._state,
                    self.config["fully_connected_layer_1_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_layer_2 = tf.layers.dense(
                    phi_state_layer_1,
                    self.config["fully_connected_layer_2_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_layer_3 = tf.layers.dense(
                    phi_state_layer_2,
                    self.config["fully_connected_layer_3_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu)
                phi_state_layer_4 = tf.layers.dense(
                    phi_state_layer_3,
                    self.config["fully_connected_layer_4_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu)

            self.q_values_predict = tf.layers.dense(
                phi_state_layer_4,
                self.config["action_space"],
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='Q_predict')

            # tf.summary.histogram('q_values_predict', self.q_values_predict)

            with tf.variable_scope('q_predict'):
                # size of q_value_predict is [BATCH_SIZE, 1]
                action_indices = tf.stack(
                    [
                        tf.range(tf.shape(self._action)[0]),
                        self._action
                    ],
                    axis=1
                )
                self.q_value_predict = tf.gather_nd(self.q_values_predict, action_indices)
                self.action_output = tf.argmax(self.q_values_predict)

        with tf.variable_scope('target_net_' + self.aid):
            # batch_norm_next_state = tf.contrib.layers.batch_norm(self._next_state)

            with tf.variable_scope('phi_net'):
                phi_state_next_layer_1 = tf.layers.dense(
                    self._next_state,
                    self.config["fully_connected_layer_1_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_next_layer_2 = tf.layers.dense(
                    phi_state_next_layer_1,
                    self.config["fully_connected_layer_2_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_next_layer_3 = tf.layers.dense(
                    phi_state_next_layer_2,
                    self.config["fully_connected_layer_3_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )
                phi_state_next_layer_4 = tf.layers.dense(
                    phi_state_next_layer_3,
                    self.config["fully_connected_layer_4_node_num"],
                    kernel_initializer=w_initializer,
                    bias_initializer=b_initializer,
                    activation=tf.nn.relu
                )

            self.q_values_target = tf.layers.dense(
                phi_state_next_layer_4,
                self.config["action_space"],
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='Q_target'
            )

            with tf.variable_scope('q_real'):
                # size of q_value_real is [BATCH_SIZE, 1]
                q_value_max = tf.reduce_max(self.q_values_target, axis=1)
                q_value_real = self._reward + \
                    self.config["gamma"] * q_value_max
                self.q_value_real = tf.stop_gradient(q_value_real)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(
                    self.q_value_real,
                    self.q_value_predict,
                    name='mse'
                )
            )

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                self.config["_learning_rate"]).minimize(self.loss)

        target_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope="target_net_" + self.aid
        )
        eval_params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope="eval_net_" + self.aid
        )

        with tf.variable_scope('soft_replacement'):
            self.update_q_net = [
                tf.assign(t, e)
                for t, e in zip(target_params, eval_params)
            ]

    def fit(self):
        # sample batch memory from all memory
        if self.buffer_count > self.config["memory_size"]:
            sample_indices = np.random.choice(
                self.config["memory_size"],
                size=self.config["batch_size"]
            )
        else:
            sample_indices = np.random.choice(
                self.buffer_count,
                size=self.config["batch_size"]
            )

        batch = self.buffer[sample_indices, :]
        batch_s = batch[:, :self.config["state_space"]]
        batch_a = batch[:, self.config["state_space"]]
        batch_r = batch[:, self.config["state_space"] + 1]
        batch_s_n = batch[:, -self.config["state_space"]:]

        _, cost = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self._state: batch_s,
                self._action: batch_a,
                self._reward: batch_r,
                self._next_state: batch_s_n
            }
        )

        self.epsilon -= self.config["epsilon_decay"]
        # self.writer.add_summary(summaries, global_step)

        if not self.step_counter % 100:
            self.update_q()

    def save_transition(self, state, action, reward, state_next):
        """
        Save transition to buffer
        """
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_count % self.config["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_count += 1

    def update_q(self):
        """
        Copy weights from eval_net to target_net
        """
        self.sess.run(self.update_q_net)

    def predict(self, is_explore, **kwargs):
        """
        Choose an action
        """
        if not is_explore or random.random() >= self.epsilon:
            return np.argmax(self.sess.run(self.action_output,
                                           feed_dict={self._state: kwargs["state"]}))
        else:
            return np.random.randint(self.config["action_space"])


if __name__ == "__main__":

    # -------------- parameters initialize --------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=1)
    parser.add_argument('--sustainable_weight', type=float, default=0.5)
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--replenishment_rate', type=float, default=0.5)
    parsed_args = parser.parse_args()

    conf = json.load(open('../config.json', 'r'))
    training_conf = conf["training_config"]
    training_conf["num_agents"] = parsed_args.n_agents

    env_conf = conf["env_config"]
    env_conf["sustain_weight"] = parsed_args.sustainable_weight
    env_conf["replenishment_rate"] = parsed_args.replenishment_rate
    env = environment.GameEnv(env_conf)

    dir_conf, agent_config = conf["dir_config"], conf["dqn"]
    dir_conf["model_save_path"] = dir_conf["model_save_path"] + '_' + \
                                  str(env_conf["sustain_weight"]) + '_' + \
                                  str(training_conf["num_agents"]) + '/'

    agent_list = []
    for i in range(training_conf["num_agents"]):
        agt = DqnAgent("DQN_" + str(i), agent_config)
        agent_list.append(agt)

    # -------------- start train mode --------------

    for t in range(training_conf["num_trial"]):

        curr_version = 'v_' + str(t)
        RESULT_PATH = dir_conf["model_save_path"] + 'dqn_results/' + curr_version + '/'
        MODEL_PATH = dir_conf["model_save_path"] + curr_version + '/'

        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)

        avg_scores = []
        global_step = 0
        phi_state = [np.zeros((agent_config["time_steps"], agent_config["state_space"])) for _ in
                     range(training_conf["num_agents"])]

        for agt in agent_list:
            agt.start(learning_mode=True)

        for epoch in range(training_conf["train_epochs"]):

            if agent_list[0].epsilon <= agent_config["min_epsilon"]:
                break

            env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf["num_agents"]

            score = 0

            for time in range(training_conf["max_round"]):
                # actions -> [Increase effort, Decrease effort, IDLE]
                actions = [0] * training_conf["num_agents"]

                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(np.mean(phi_state[index], axis=0), axis=0))
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
                    phi_curr_state = np.mean(phi_state[index], axis=0)
                    phi_state[index][global_step % agent_config["time_steps"], :] = np.asarray(next_states[index])
                    phi_next_state = np.mean(phi_state[index], axis=0)

                    player.save_transition(phi_curr_state, actions[index], rewards[index], phi_next_state)

                global_step += 1

                if done:
                    break

            if not epoch % 2:
                [player.learn(global_step) for player in agent_list]

            score /= training_conf["num_agents"]

            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(epoch, training_conf["train_epochs"], score, agent_list[0].epsilon))

            avg_scores.append(score)

        for a in agent_list:
            a.save(dir_path=MODEL_PATH)
            a.sess.close()

        # -------------- save results --------------

        plt.switch_backend('agg')
        plt.plot(avg_scores)
        plt.interactive(False)
        plt.xlabel('Epoch')
        plt.ylabel('Avg score')
        plt.savefig(dir_conf["model_save_path"] + 'dqn_training_plot')

        with open(RESULT_PATH + 'train_avg_score.txt', "w+") as f:
            for r in avg_scores:
                f.write(str(r) + '\n')

        # -------------- start test mode --------------

        for agt in agent_list:
            agt.start(learning_mode=False, dir_path=MODEL_PATH)

        avg_assets = [0]
        resource_level = []
        for epoch in range(1):
            # state -> [X, Pi]
            env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf["num_agents"]

            score = 0

            for time in range(training_conf["max_round"]):
                resource_level.append(env.common_resource_pool)

                # actions -> [Increase effort, Decrease effort, IDLE]
                actions = [0] * training_conf["num_agents"]

                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(np.mean(phi_state[index], axis=0), axis=0))
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
                score /= training_conf["num_agents"]
                avg_scores.append(score)
                avg_assets.append(avg_assets[-1] + next_states[0][3] / training_conf["num_agents"])

                for index, player in enumerate(agent_list):
                    phi_state[index][global_step % agent_config["time_steps"], :] = np.asarray(next_states[index])

                global_step += 1

                if done:
                    break

            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(epoch, training_conf["test_epochs"], score, agent_list[0].epsilon))

        for a in agent_list:
            a.sess.close()

        # -------------- save results --------------

        plt.switch_backend('agg')
        plt.plot(avg_scores)
        plt.interactive(False)
        plt.xlabel('Epoch')
        plt.ylabel('Avg score')
        plt.savefig(dir_conf["model_save_path"] + 'dqn_test_plot')

        with open(RESULT_PATH + 'test_avg_score.txt', "w+") as f:
            for s in avg_scores:
                f.write(str(s) + '\n')

        with open(RESULT_PATH + 'test_assets.txt', "w+") as f:
            for a in avg_assets:
                f.write(str(a) + '\n')

        with open(RESULT_PATH + "test_resource_level.txt", "w+") as f:
            for r in resource_level:
                f.write(str(r) + '\n')
