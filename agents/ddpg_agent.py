# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 下午3:48
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : ddpg_agent.py
# @Software: PyCharm Community Edition


import base_agent
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import sys

sys.path.append("../")
import environment


class DDPGAgent(base_agent.BaseAgent):
    def __init__(self, name, opt, learning_mode=True):
        super().__init__(name, opt, learning_mode)
        self.buffer = np.zeros((self.opt["memory_size"], self.opt["state_space"] * 2 + self.opt["action_space"] + 1))
        self.buffer_item_count = 0

    def _build_model(self):
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
            batch_norm_state = tf.contrib.layers.batch_norm(state)
            phi_state_layer_1 = tf.layers.dense(batch_norm_state,
                                                self.opt["fully_connected_layer_1_node_num"],
                                                tf.nn.relu,
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init,
                                                trainable=trainable)

            phi_state_layer_2 = tf.layers.dense(phi_state_layer_1,
                                                self.opt["fully_connected_layer_2_node_num"],
                                                tf.nn.relu,
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init,
                                                trainable=trainable)

            phi_state_layer_3 = tf.layers.dense(phi_state_layer_2,
                                                self.opt["fully_connected_layer_3_node_num"],
                                                tf.nn.relu,
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init,
                                                trainable=trainable)

            action_prob = tf.layers.dense(phi_state_layer_3,
                                          self.opt["action_space"],
                                          tf.nn.sigmoid,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init,
                                          trainable=trainable)

            return tf.multiply(action_prob, self.opt["action_upper_bound"])

    @staticmethod
    def __build_critic(state, action, scope, trainable=True):
        w_init, b_init = tf.random_normal_initializer(.0, .001), tf.constant_initializer(.001)

        with tf.variable_scope(scope):
            batch_norm_state = tf.contrib.layers.batch_norm(state)
            phi_state = tf.layers.dense(batch_norm_state,
                                        32,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        trainable=trainable)

            batch_norm_action = tf.layers.batch_normalization(action)
            phi_action = tf.layers.dense(batch_norm_action,
                                         32,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init,
                                         trainable=trainable)

            phi_state_action = tf.layers.dense(tf.nn.relu(phi_state + phi_action),
                                               32,
                                               kernel_initializer=w_init,
                                               bias_initializer=b_init,
                                               trainable=trainable)

            q_value = tf.layers.dense(phi_state_action,
                                      1,
                                      kernel_initializer=w_init,
                                      bias_initializer=b_init,
                                      trainable=trainable)

            return q_value

    def save_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, action, [reward], state_next))
        index = self.buffer_item_count % self.opt["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_item_count += 1

    def get_sample_batch(self):
        indices = np.random.choice(self.opt["memory_size"], size=self.opt["batch_size"])
        batch = self.buffer[indices, :]
        state = batch[:, :self.opt["state_space"]]
        action = batch[:, self.opt["state_space"]: self.opt["state_space"] + self.opt["action_space"]]
        reward = batch[:, -self.opt["state_space"] - 1: -self.opt["state_space"]]
        state_next = batch[:, -self.opt["state_space"]:]
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

    def choose_action(self, state, upper_bound):
        action = self.sess.run(self.a_predict, {self._state: state})
        exploration_scale = 1000 * self.epsilon
        action = np.clip(np.random.normal(action[0], exploration_scale), 0, upper_bound)
        return action[0]


if __name__ == "__main__":

    # -------------- parameters initialize --------------

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

    dir_conf, agent_opt = conf["dir_config"], conf["ddpg"]
    dir_conf["model_save_path"] = dir_conf["model_save_path"] + '_' + \
                                  str(env_conf["sustain_weight"]) + '_' + \
                                  str(training_conf["num_agents"]) + '/'

    avg_scores = []
    global_step = 0
    phi_state = [np.zeros((agent_opt["time_steps"], agent_opt["state_space"])) for _ in range(training_conf["num_agents"])]

    # -------------- start train mode --------------

    if not parsed_args.is_test:

        agent_list = []
        for i in range(training_conf["num_agents"]):
            player = DDPGAgent("DDPG_" + str(i), agent_opt)
            player.start(dir_path=dir_conf["model_save_path"])
            agent_list.append(player)

        for epoch in range(training_conf["train_epochs"]):
            if agent_list[0].epsilon <= agent_opt["min_epsilon"]:
                break

            env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf[
                "num_agents"]
            score = 0

            for time in range(training_conf["max_round"]):
                actions = [0] * training_conf["num_agents"]
                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(np.mean(phi_state[index], axis=0), axis=0),
                                                  env.common_resource_pool / training_conf["num_agents"])
                    if action < agent_opt["action_lower_bound"]:
                        action = agent_opt["action_lower_bound"]
                    efforts[index] = action

                next_states, rewards, done = env.step(efforts)
                
                score += sum(rewards)

                for index, player in enumerate(agent_list):
                    phi_curr_state = np.mean(phi_state[index], axis=0)
                    phi_state[index][global_step % agent_opt["time_steps"], :] = np.asarray(next_states[index])
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
            a.save(dir_path=dir_conf["model_save_path"])
            a.sess.close()

        # -------------- save results --------------

        plt.switch_backend('agg')
        plt.plot(avg_scores)
        plt.interactive(False)
        plt.xlabel('Epoch')
        plt.ylabel('Avg score')
        plt.savefig(dir_conf["model_save_path"] + 'ddpg/training_plot')

        with open(dir_conf["model_save_path"] + 'ddpg/avg_score.txt', "w+") as f:
            for r in avg_scores:
                f.write(str(r) + '\n')

    # -------------- start test mode --------------

    else:

        agent_list = []
        for i in range(training_conf["num_agents"]):
            player = DDPGAgent("DDPG_" + str(i), agent_opt, learning_mode=False)
            player.start(dir_path=dir_conf["model_save_path"])
            agent_list.append(player)
        
        avg_assets = [0]
        resource_level = []
        for epoch in range(1):

            env.reset()
            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf[
                "num_agents"]
            avg_scores = []

            for time in range(training_conf["test_max_round"]):
                resource_level.append(env.common_resource_pool)

                actions = [0] * training_conf["num_agents"]
                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(np.mean(phi_state[index], axis=0), axis=0),
                                                  env.common_resource_pool / training_conf["num_agents"])
                    if action < agent_opt["action_lower_bound"]:
                        action = agent_opt["action_lower_bound"]
                    efforts[index] = action

                next_states, rewards, done = env.step(efforts)

                avg_scores.append(sum(rewards) / training_conf["num_agents"])
                avg_assets.append(next_states[3] / training_conf["num_agents"]) 

                for index, player in enumerate(agent_list):
                    phi_state[index][global_step % agent_opt["time_steps"], :] = np.asarray(next_states[index])

                global_step += 1

                if done:
                    break

        for a in agent_list:
            a.sess.close()

        # -------------- save results --------------

        with open(dir_conf["model_save_path"] + 'ddpg/avg_scores.txt', "w+") as f:
            for s in avg_scores:
                f.write(str(s) + '\n')

        with open(dir_conf["model_save_path"] + 'ddpg/test_assets.txt', "w+") as f:
            for a in avg_assets:
                f.write(str(a) + '\n')

        with open(dir_conf["model_save_path"] + "ddpg/test_resource_level.txt", "w+") as f:
            for r in resource_level:
                f.write(str(r) + '\n')
