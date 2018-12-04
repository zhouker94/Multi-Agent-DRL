import tensorflow as tf
from abc import abstractmethod


class BaseModel(object):
    def __init__(self, aid, config):
        self.aid = aid
        self.config = config
        self.step_counter = 0
        self.epsilon = self.config["init_epsilon"]

        # input & output
        self._state = tf.placeholder(
            tf.float32,
            shape=[None, self.config["state_space"]],
            name='state'
        )
        self._next_state = tf.placeholder(
            tf.float32,
            shape=[None, self.config["state_space"]],
            name='next_state'
        )
        self._reward = tf.placeholder(
            tf.float32,
            shape=[None, 1],
            name='reward'
        )
        self._action = tf.placeholder(
            tf.float32,
            [None, 1],
            name='action'
        )
        self._dropout_keep_prob = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name='dropout_keep_prob'
        )

        self._build_graph()
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    @abstractmethod
    def _build_graph(self):
        """
        Build tensorflow nn model.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        train nn
        """
        pass

    @abstractmethod
    def predict(self, state, is_explore, **kwargs):
        pass

    @abstractmethod
    def save_transition(self, state, action, reward, next_state):
        pass

    def save_model(self, path):
        """
        Save Tensorflow model
        """
        return self.saver.save(self.sess, path)

    def close_model(self):
        self.sess.close()
