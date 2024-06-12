from tensorflow import compat as ttf
from abc import abstractmethod

tf = ttf.v1
tf.disable_eager_execution()


class BaseModel(object):
    def __init__(self, model_id, config, ckpt_path):
        self.model_id = model_id
        self.config = config
        self.step_counter = 0

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
        self._dropout_keep_prob = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name='dropout_keep_prob'
        )

        self._build_graph()
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if ckpt_path:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print("Cannot restore model, does not exist")
                raise Exception
        else:
            self.sess.run(tf.global_variables_initializer())

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
    def predict(self, state, epsilon, **kwargs):
        pass

    @abstractmethod
    def save_transition(self, state, action, reward, next_state):
        pass

    def save_model(self, path):
        """
        Save Tensorflow model
        """
        return self.saver.save(self.sess, path)

    def close(self):
        self.sess.close()
