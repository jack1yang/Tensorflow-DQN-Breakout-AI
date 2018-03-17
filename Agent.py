import DQN
import random
import numpy as np


class Agent_RL:
    def __init__(self, restore=False):
        sess_conf = DQN.tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        self.sess = DQN.tf.Session(config=sess_conf)

        self.Q_main = DQN.DQN(self.sess, name="main")
        self.Q_target = DQN.DQN(self.sess, name="target")

        self.sess.run(DQN.tf.global_variables_initializer())
        self.copy_ops = DQN.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        self.copy()

        if restore:
            self.restore()
            self.copy()

    def policy(self, state, epsilon=0.1):
        if np.random.rand(1) < epsilon:
            return random.choice(range(4))
        else:
            return np.argmax(self.Q_main.predict(state))

    def learning_rate(self, step):
        return self.Q_main.learning_rate(step)

    def save(self):
        self.Q_main.save()

    def restore(self):
        self.Q_main.restore()

    def copy(self):
        self.sess.run(self.copy_ops)
