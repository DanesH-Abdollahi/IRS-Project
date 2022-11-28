import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from math import pi


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=256, name="Critic", chkpt_dir="../tmp/ddpg"):
        super().__init__()
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5")

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=256, n_actions=2, name="Actor", chkpt_dir="../tmp/ddpg"):
        super().__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5")

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='softsign')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob) * pi

        return mu
