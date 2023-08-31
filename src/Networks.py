import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from math import pi


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name="Critic",
                 chkpt_dir="../tmp/td3"):
        super().__init__()
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_td3")

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        inputs = tf.concat([state, action], axis=1)
        value = self.fc1(inputs)
        value = self.fc2(value)
        q = self.q(value)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions,
                 bound, name="Actor", chkpt_dir="../tmp/td3"):
        super().__init__()
        self.n_actions = n_actions
        self.bound = bound
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_td3")

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions - 1, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob) * self.bound

        return mu


class PowerActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, num_of_users, name="PowerActor", chkpt_dir="../tmp/td3"):
        super().__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_td3")

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(num_of_users - 1, activation='tanh')

    def call(self, state):
        power = self.fc1(state)
        power = self.fc2(power)

        return power
