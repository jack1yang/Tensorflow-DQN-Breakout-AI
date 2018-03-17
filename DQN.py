import tensorflow as tf
import numpy as np

# input/output
input_size = 84, 84
history_size = 4
output_size = 4

# optimizer parameter
learning_rate = 0.0025
gradient_momentum = 0.95
learning_rate_minimum = 0.00025
learning_rate_decay = 0.96
learning_rate_decay_step = 50000


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


class DQN:
    def __init__(self, sess, name="main"):
        self.session = sess
        self.net_name = name

        self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.net_name):
            initializer = tf.truncated_normal_initializer(0, 0.02)

            self._X = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="input_x")
            self._Y = tf.placeholder(shape=[None, 4], dtype=tf.float32)

            l1 = tf.layers.conv2d(inputs=self._X, filters=32, kernel_size=[8, 8], strides=4, activation=tf.nn.relu,
                                  kernel_initializer=initializer)
            l2 = tf.layers.conv2d(inputs=l1, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu,
                                  kernel_initializer=initializer)
            l3 = tf.layers.conv2d(inputs=l2, filters=64, kernel_size=[3, 3], strides=1, activation=tf.nn.relu,
                                  kernel_initializer=initializer)
            l4 = tf.reshape(l3, [-1, 7 * 7 * 64])
            l5 = tf.layers.dense(inputs=l4, units=512, activation=tf.nn.relu)

            self._Qpred = tf.layers.dense(inputs=l5, units=4)
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(learning_rate_minimum,
                                               tf.train.exponential_decay(learning_rate,
                                                                          self.learning_rate_step,
                                                                          learning_rate_decay_step,
                                                                          learning_rate_decay,
                                                                          staircase=True))
            self._train = tf.train.RMSPropOptimizer(self.learning_rate_op,
                                                    momentum=gradient_momentum,
                                                    epsilon=0.01).minimize(self._loss)

            self.saver = tf.train.Saver()

    def predict(self, state):
        _X = np.reshape(state, [1, 84, 84, 4])
        return self.session.run(self._Qpred, feed_dict={self._X: _X})

    def update(self, state, values, step):
        return self.session.run(self._train, feed_dict={self._X: state, self._Y: values, self.learning_rate_step: step})

    def learning_rate(self, step):
        return self.session.run(self.learning_rate_op, feed_dict={self.learning_rate_step: step})

    def save(self):
        save_path = self.saver.save(self.session, "./data/model.ckpt")
        print("Model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.session, "./data/model.ckpt")
        print("Model is restored.")
