import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from math import pi


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name="Critic",
                 chkpt_dir="../tmp/ddpg"):
        super().__init__()
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5")

        self.bn0 = BatchNormalization()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.bn1 = BatchNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.bn2 = BatchNormalization()

        self.action_value = Dense(self.fc2_dims, activation='relu')

        self.concat = tf.keras.layers.Concatenate()
        self.output_layer1 = Dense(self.fc1_dims, activation="relu")
        self.output_layer2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        state_value = self.bn0(state)
        state_value = self.fc1(state_value)
        state_value = self.bn1(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)

        action_value = self.concat([state_value, action_value])
        action_value = self.output_layer1(action_value)
        action_value = self.output_layer2(action_value)
        q = self.q(action_value)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions,
                 bound, name="Actor", chkpt_dir="../tmp/ddpg"):
        super().__init__()
        self.n_actions = n_actions
        self.bound = bound
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5")

        self.bn0 = BatchNormalization()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.bn1 = BatchNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.bn2 = BatchNormalization()
        self.mu = Dense(self.n_actions, activation='sigmoid')

    def call(self, state):
        prob = self.bn0(state)
        prob = self.fc1(prob)
        prob = self.bn1(prob)
        prob = self.fc2(prob)
        prob = self.bn2(prob)
        mu = self.mu(prob)
        return mu * self.bound
