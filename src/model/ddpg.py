from model import base_model
import tensorflow as tf
import numpy as np
from tensorflow import compat as ttf

tf = ttf.v1
tf.disable_eager_execution()


class DDPGModel(base_model.BaseModel):
    def __init__(self, aid, config, ckpt_path):
        super().__init__(aid, config, ckpt_path)
        self.obs = np.zeros(
            (
                self.config["time_steps"],
                self.config["state_space"]
            )
        )
        self.buffer = np.zeros(
            (
                self.config["memory_size"],
                self.config["state_space"] * 2 +
                self.config["action_space"] + 1
            )
        )
        self.buffer_counter = 0

    def _build_graph(self):
        self._reward = tf.placeholder(
            tf.float32,
            shape=[None, 1],
            name='reward'
        )
        self._action = tf.placeholder(
            tf.float32,
            [None, self.config["action_space"]],
            name='action'
        )
        self._phase = tf.placeholder(tf.bool, name='phase')
        self._reward = tf.placeholder(
            tf.float32,
            [None, 1],
            name='input_reward'
        )
        self.tau = tf.constant(self.config["tau"])
        self.a_predict = self.__build_actor_nn(
            self._state,
            "predict/actor" + self.model_id,
            self._phase,
            trainable=True
        )
        self.a_next = self.__build_actor_nn(
            self._next_state,
            "target/actor" + self.model_id,
            self._phase,
            trainable=False
        )
        self.q_predict = self.__build_critic(
            self._state,
            self.a_predict,
            "predict/critic" + self.model_id,
            self._phase,
            trainable=True
        )
        self.q_next = self.__build_critic(
            self._next_state,
            self.a_next,
            "target/critic" + self.model_id,
            self._phase,
            trainable=False
        )
        self.params = []

        for scope in ['predict/actor' + self.model_id,
                      'target/actor' + self.model_id,
                      'predict/critic' + self.model_id,
                      'target/critic' + self.model_id]:
            self.params.append(
                tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope
                )
            )

        self.actor_loss = -tf.reduce_mean(self.q_predict)
        self.actor_train_op = tf.train.AdamOptimizer(
            self.config["learning_rate"]
        ).minimize(
            self.actor_loss,
            var_list=self.params[0]
        )

        self.q_target = self._reward + self.config['gamma'] * self.q_next
        self.critic_loss = tf.losses.mean_squared_error(
            self.q_target, self.q_predict)
        self.critic_train_op = tf.train.AdamOptimizer(
            self.config["learning_rate"] * 2
        ).minimize(
            self.critic_loss,
            var_list=self.params[2]
        )

        self.update_actor = [
            tf.assign(t_a, (1 - self.tau) * t_a + self.tau * p_a)
            for p_a, t_a in zip(self.params[0], self.params[1])
        ]
        self.update_critic = [
            tf.assign(t_c, (1 - self.tau) * t_c + self.tau * p_c)
            for p_c, t_c in zip(self.params[2], self.params[3])
        ]

    def __build_actor_nn(self, state, scope, phase, trainable=True):
        w_init, b_init = \
            tf.random_normal_initializer(.0, .1), tf.constant_initializer(.1)

        with tf.variable_scope(scope):
            # batch_norm_state = tf.layers.batch_normalization(state, axis=0)
            # batch_norm_state = tf.contrib.layers.batch_norm(
            #     state, center=True, scale=True, is_training=phase)

            phi_state_layer_1 = tf.keras.layers.Dense(
                self.config["fully_connected_layer_1_node_num"],
                activation=tf.nn.relu,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(state)

            phi_state_layer_2 = tf.keras.layers.Dense(
                self.config["fully_connected_layer_2_node_num"],
                activation=tf.nn.relu,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(phi_state_layer_1)

            phi_state_layer_3 = tf.keras.layers.Dense(
                self.config["fully_connected_layer_3_node_num"],
                activation=tf.nn.relu,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(phi_state_layer_2)

            action_prob = tf.keras.layers.Dense(
                self.config["action_space"],
                activation=tf.nn.sigmoid,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(phi_state_layer_3)

            return tf.multiply(
                action_prob, self.config["action_upper_bound"]
            )

    @staticmethod
    def __build_critic(state, action, scope, phase, trainable=True):
        w_init = tf.random_normal_initializer(.0, .1)
        b_init = tf.constant_initializer(.1)
        with tf.variable_scope(scope):
            # batch_norm_state = tf.contrib.layers.batch_norm(
            #     state, center=True, scale=True, is_training=phase,
            #     trainable=trainable)
            phi_state = tf.keras.layers.Dense(
                32,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(state)
            # batch_norm_action = tf.contrib.layers.batch_norm(
            #     action, center=True, scale=True, is_training=phase,
            #     trainable=trainable)
            phi_action = tf.keras.layers.Dense(
                32,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(action)

            phi_state_action = tf.keras.layers.Dense(
                32,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(tf.nn.relu(phi_state + phi_action))

            q_value = tf.keras.layers.Dense(
                1,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                trainable=trainable
            )(phi_state_action)

            return q_value

    def save_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, action, [reward], state_next))
        index = self.buffer_counter % self.config["memory_size"]
        self.buffer[index, :] = transition
        self.buffer_counter += 1

    def get_sample_batch(self):
        indices = np.random.choice(
            self.config["memory_size"],
            size=self.config["batch_size"]
        )
        batch = self.buffer[indices, :]
        state = batch[:, :self.config["state_space"]]
        action = batch[
                 :,
                 self.config["state_space"]: self.config["state_space"] +
                                             self.config["action_space"]
                 ]
        reward = batch[:, -self.config["state_space"] -
                          1: -self.config["state_space"]]
        state_next = batch[:, -self.config["state_space"]:]
        return state, action, reward, state_next

    def fit(self):
        self.sess.run([self.update_actor, self.update_critic])
        state, action, reward, state_next = self.get_sample_batch()
        self.sess.run(
            self.actor_train_op,
            {self._phase: True,
             self._state: state}
        )
        self.sess.run(self.critic_train_op, {
            self._phase: True,
            self._state: state,
            self.a_predict: action,
            self._reward: reward,
            self._next_state: state_next
        })

    def predict(self, state, epsilon, **kwargs):
        action = self.sess.run(
            self.a_predict,
            feed_dict={
                self._state: state,
                self._phase: False}
        )
        exploration_scale = 1000 * epsilon
        action = np.clip(
            np.random.normal(action[0], exploration_scale),
            0,
            kwargs["upper_bound"]
        )
        return action[0]
