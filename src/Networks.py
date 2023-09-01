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
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.bn1 = BatchNormalization()

        # self.action_value_1 = Dense(1024, activation='relu')
        self.action_value_2 = Dense(256, activation='relu')
        self.action_value_3 = Dense(128, activation='relu')
        self.action_value_4 = Dense(64, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        # self.output_layer1 = Dense(self.fc1_dims, activation="relu")
        # self.output_layer2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        # combined = self.concat([state, action])
        state_value = self.bn0(action)
        state_value = self.fc1(state_value)
        state_value = self.fc2(state_value)
        combined = self.concat([state_value, state])
        combined = self.bn1(combined)
        # action_value = self.action_value_1(combined)
        action_value = self.action_value_2(combined)
        action_value = self.action_value_3(action_value)
        action_value = self.action_value_4(action_value)
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
        # self.fc01 = Dense(3010, activation='relu')
        self.fc0 = Dense(2024, activation='relu')
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.fc3 = Dense(256, activation='relu')
        self.fc4 = Dense(64, activation='relu')
        # self.fc4 = Dense(64, activation='relu')
        self.mu = Dense(self.n_actions - 1, activation='sigmoid')

    def call(self, state):
        prob = self.bn0(state)
        # prob = self.fc01(prob)
        prob = self.fc0(prob)
        prob = self.fc1(prob)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)
        mu = self.mu(prob) * self.bound

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
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(num_of_users - 1, activation='sigmoid')

    def call(self, state):
        power = self.bn0(state)
        power = self.fc1(power)
        power = self.fc2(power)
        return power
