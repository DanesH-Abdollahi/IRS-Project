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
        # self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.bn2 = BatchNormalization()

        self.action_value = Dense(self.fc1_dims, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        # self.output_layer1 = Dense(self.fc1_dims, activation="relu")
        self.output_layer2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        state_value = self.bn0(state)
        state_value = self.fc1(state_value)
        # state_value = self.fc2(state_value)
        # state_value = self.bn2(state_value)

        action_value = self.bn1(action)
        action_value = self.action_value(action_value)
        q = self.concat([state_value, action_value])
        q = self.bn2(q)
        # q = self.output_layer1(q)
        q = self.output_layer2(q)
        q = self.q(q)
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
        # self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(self.fc2_dims, activation='relu')
        self.fc4 = Dense(64, activation='relu')
        self.mu = Dense(self.n_actions - 1, activation='sigmoid')
        # self.concat = tf.keras.layers.Concatenate()
        # self.power1 = Dense(128, activation="sigmoid")
        # self.power2 = Dense(64, activation="sigmoid")
        # self.power3 = Dense(32, activation="sigmoid")
        # self.power4 = Dense(1, activation="sigmoid")

    def call(self, state):
        prob = self.bn0(state)
        # power = self.power1(prob)

        prob = self.fc1(prob)
        # prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)

        mu = self.mu(prob) * self.bound
        # power = self.power2(power)
        # power = self.power3(power)
        # power = self.power4(power)

        # out = self.concat([mu, power])
        return mu


class PowerActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, num_of_users, name="PowerActor", chkpt_dir="../tmp/ddpg"):
        super().__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5")

        self.bn0 = BatchNormalization()
        # self.fc1 = Dense(1024, activation='relu')
        # self.fc2 = Dense(512, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        # self.fc4 = Dense(64, activation='relu')
        self.fc5 = Dense(num_of_users - 1, activation='sigmoid')

    def call(self, state):
        power = self.bn0(state)
        # power = self.fc1(power)
        # power = self.fc2(power)
        power = self.fc3(power)
        # power = self.fc4(power)
        power = self.fc5(power)
        return power
