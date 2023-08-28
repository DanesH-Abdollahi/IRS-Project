import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization
from math import pi


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, name="Critic"):
        super().__init__()
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # self.bn0 = BatchNormalization()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, name="Actor"):
        super().__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        # self.bn0 = BatchNormalization()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(self.n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
