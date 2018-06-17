#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/18 0:18
# @Author  : Hanwei Zhu
# @File    : drqn_agent.py


"""
class DrqnAgent(BaseAgent):
    def __init__(self, name, epsilon=const.EPSILON_INIT, learning_rate=0.01, learning_mode=True):
        super().__init__(name, epsilon, learning_rate, learning_mode)
        self._target_q = drq_network.TargetDRQNetwork(scope=const.TARGET_Q_SCOPE + name,
                                                      inputs=(self._input_x, self._input_y),
                                                      dropout_keep_prob=self._dropout_keep_prob)
        self._online_q = drq_network.OnlineDRQNetwork(scope=const.ONLINE_Q_SCOPE + name,
                                                      inputs=(self._input_x, self._input_y),
                                                      dropout_keep_prob=self._dropout_keep_prob)
        self._target_q.define_graph()
        self._online_q.define_graph()

        target_q_weights = self._target_q.layered_cell.trainable_variables
        online_q_weights = self._online_q.layered_cell.trainable_variables

        self.replace_target_op = \
            [tf.assign(t, o) for t, o in zip(target_q_weights, online_q_weights)]

        # The rl_brain should hold a tf session
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        # train_writer = tf.summary.FileWriter(const.LOG_PATH, sess.graph)
        if os.path.exists(const.MODEL_SAVE_PATH):
            print("load successfully")
            self.saver.restore(self.sess, const.MODEL_SAVE_PATH + self._name)
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
        np.random.seed(seed=hash(self._name) % 256)

    def choose_action(self, state):
        q_values = self.sess.run(self._online_q.outputs[const.Q_VALUE_OUTPUT],
                                 feed_dict={self._input_x: state,
                                            self._dropout_keep_prob: 0.5})

        if not self._learning_mode or random.random() >= self.epsilon:
            return np.argmax(q_values[:, -1, :], axis=1)
        else:
            return np.random.randint(const.ACTION_SPACE)

    def learn(self):
        # sample batch memory from all memory
        (states, actions, rewards, terminate) = random.sample(self.memory, 1)[0]
        episode_length = len(actions)

        # current states
        q_eval = self.sess.run(
            self._online_q.outputs[const.Q_VALUE_OUTPUT],
            feed_dict={
                self._input_x: states,
                self._dropout_keep_prob: 1
            })
        # next states
        q_next = self.sess.run(
            self._target_q.outputs[const.MAX_Q_OUTPUT],
            feed_dict={
                self._input_x: states[:, 1:, :],
                self._dropout_keep_prob: 1
            })

        targets = np.zeros(q_eval.shape)

        # Bootstrapped Sequential Updates
        for index in range(episode_length):

            target = rewards[index]
            if not terminate[index]:
                target = target + self.gamma * q_next[:, index]

            target_f = q_eval[:, index, :]
            target_f[0][actions[index]] = target
            targets[:, index, :] = target_f

        _, loss = self.sess.run([self._online_q.outputs[const.ADAM_OPTIMIZER],
                                 self._online_q.outputs[const.REDUCE_MEAN_LOSS]],
                                feed_dict={self._input_x: states,
                                           self._input_y: targets,
                                           self._dropout_keep_prob: 0.8})

        self.epsilon -= const.EPSILON_DECAY

    def store_experience(self, state, actions, rewards, terminate):
        transition = (state, actions, rewards, terminate)
        self.memory.append(transition)
"""
