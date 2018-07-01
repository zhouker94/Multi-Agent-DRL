# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 下午3:48
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : ddpg_agent.py
# @Software: PyCharm Community Edition


from agents import base_agent
import tensorflow as tf
import numpy as np


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

        for scope in ['predict/actor'+ self._name,
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
            phi_state = tf.layers.dense(state,
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

    @staticmethod
    def __build_critic(state, action, scope, trainable=True):
        w_init, b_init = tf.random_normal_initializer(.0, .001), tf.constant_initializer(.001)

        with tf.variable_scope(scope):
            phi_state = tf.layers.dense(state,
                                        32,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        trainable=trainable)

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

    def choose_action(self, state, action_upper_bound):
        action = self.sess.run(self.a_predict, {self._state: state})
        exploration_scale = 100 * self.epsilon
        action = np.clip(np.random.normal(action[0], exploration_scale), 0, action_upper_bound)
        return action[0]
