import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, PReLU, LeakyReLU
from tensorflow.keras.initializers import HeUniform, GlorotNormal, HeNormal
from numpy import array


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name="Critic", chkpt_dir="../tmp/ddpg"):
        super().__init__()
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5"
        )

        self.initilizer = HeUniform(seed=23)
        self.initilizer1 = GlorotNormal(seed=23)
        self.initilizer2 = HeNormal(seed=23)

        self.activation00 = PReLU()
        self.activation0 = PReLU()
        self.activation1 = PReLU()
        self.activation2 = PReLU()
        self.activation3 = PReLU()

        # self.bn0 = BatchNormalization()
        # self.fc00 = Dense(4048, activation="relu")

        # self.fc00 = Dense(
        #     2024,
        #     activation=self.activation00,
        # )

        self.fc_0 = Dense(
            1024,
            activation=self.activation0,
        )

        self.fc_1 = Dense(
            512,
            activation=self.activation1,
        )

        # self.fc0 = Dense(1024, activation="relu")
        # self.fc1 = Dense(2024, activation="relu")

        self.fc2 = Dense(
            512,
            activation=self.activation2,
        )

        # self.fc3 = Dense(
        #     256,
        #     activation=self.activation3,
        # )

        # self.bn1 = BatchNormalization()

        # self.action_value_1 = Dense(512, activation="relu")
        # self.action_value_2 = Dense(256, activation='relu')
        # self.action_value_3 = Dense(
        #     256, activation="relu", kernel_initializer=self.initilizer
        # )
        # self.action_value_4 = Dense(64, activation="relu")
        # self.concat = tf.keras.layers.Concatenate()
        # self.output_layer1 = Dense(self.fc1_dims, activation="relu")
        # self.output_layer2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        # combined = self.concat([action, state])
        state_value = action
        # state_value = self.bn0(action)
        # state_value = combined
        # state_value = self.fc00(state_value)
        state_value = self.fc_0(state_value)
        state_value = self.fc_1(state_value)
        # state_value = self.fc0(state_value)
        # state_value = self.fc1(state_value)
        state_value = self.fc2(state_value)
        # state_value = self.fc3(state_value)

        # combined = self.concat([state_value, state])

        # combined = self.bn1(combined)
        # action_value = self.action_value_1(state_value)
        # action_value = self.action_value_2(action_value)
        # action_value = self.action_value_3(state_value)
        # action_value = self.action_value_4(action_value)
        q = self.q(state_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(
        self,
        fc1_dims,
        fc2_dims,
        n_actions,
        bound,
        env,
        name="Actor",
        chkpt_dir="../tmp/ddpg",
        last_layer_activation="sigmoid",
        multi_out_layer=False,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.bound = bound
        self.model_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.last_layer_activation = last_layer_activation
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5"
        )
        self.multi_out_layer = multi_out_layer

        # self.bn0 = BatchNormalization()
        # self.fc00 = Dense(4048, activation="relu")
        # self.fc0 = Dense(2024, activation="relu")
        # self.fc2 = Dense(512, activation="relu")
        # self.fc3 = Dense(256, activation="relu")
        # self.fc4 = Dense(64, activation="relu")
        # self.fc4 = Dense(64, activation='relu')
        # self.mu = Dense(self.n_actions - 1, activation=last_layer_activation)

        if multi_out_layer:
            self.irs1_0 = Dense(256, activation="relu")
            self.irs1_0_1 = Dense(512, activation="relu")
            # self.irs1_0_2 = Dense(1024, activation="relu")
            self.irs1_1 = Dense(1024, activation="relu")
            self.irs1_2 = Dense(512, activation="relu")
            self.irs1_3 = Dense(256, activation="relu")
            # self.irs1_4 = Dense(128, activation="relu")
            self.irs1_5 = Dense(env.M1, activation=last_layer_activation)

            self.irs2_0 = Dense(256, activation="relu")
            self.irs2_0_1 = Dense(512, activation="relu")
            # self.irs2_0_2 = Dense(1024, activation="relu")
            self.irs2_1 = Dense(1024, activation="relu")
            self.irs2_2 = Dense(512, activation="relu")
            self.irs2_3 = Dense(256, activation="relu")
            # self.irs2_4 = Dense(128, activation="relu")
            self.irs2_5 = Dense(env.M2, activation=last_layer_activation)

            self.w1_0 = Dense(256, activation="relu")
            self.w1_0_1 = Dense(512, activation="relu")
            # self.w1_0_2 = Dense(1024, activation="relu")
            self.w1_1 = Dense(1024, activation="relu")
            self.w1_2 = Dense(512, activation="relu")
            self.w1_3 = Dense(256, activation="relu")
            # self.w1_4 = Dense(128, activation="relu")
            self.w1_5 = Dense(env.N, activation=last_layer_activation)

            self.w2_0 = Dense(256, activation="relu")
            self.w2_0_1 = Dense(512, activation="relu")
            # self.w2_0_2 = Dense(1024, activation="relu")
            self.w2_1 = Dense(1024, activation="relu")
            self.w2_2 = Dense(512, activation="relu")
            self.w2_3 = Dense(256, activation="relu")
            # self.w2_4 = Dense(128, activation="relu")
            self.w2_5 = Dense(env.N, activation=last_layer_activation)

            self.concat = tf.keras.layers.Concatenate(axis=1)

        else:
            self.fc1 = Dense(1024, activation="relu")
            self.fc2 = Dense(512, activation="relu")
            self.fc3 = Dense(256, activation="relu")
            self.fc4 = Dense(64, activation="relu")
            self.mu = Dense(self.n_actions - 1, activation=last_layer_activation)

    def call(self, state):
        prob = state
        # prob = self.bn0(state)
        # prob = self.fc00(prob)
        # prob = self.fc0(prob)
        # prob = self.fc1(prob)
        # prob = self.fc2(prob)
        # prob = self.fc3(prob)
        # prob = self.fc4(prob)

        if self.multi_out_layer:
            irs1 = prob
            irs1 = self.irs1_0(prob)
            irs1 = self.irs1_0_1(irs1)
            # irs1 = self.irs1_0_2(irs1)
            irs1 = self.irs1_1(irs1)
            irs1 = self.irs1_2(irs1)
            irs1 = self.irs1_3(irs1)
            # irs1 = self.irs1_4(irs1)
            irs1 = self.irs1_5(irs1)

            irs2 = prob
            irs2 = self.irs2_0(prob)
            irs2 = self.irs2_0_1(irs2)
            # irs2 = self.irs2_0_2(irs2)
            irs2 = self.irs2_1(irs2)
            irs2 = self.irs2_2(irs2)
            irs2 = self.irs2_3(irs2)
            # irs2 = self.irs2_4(irs2)
            irs2 = self.irs2_5(irs2)

            w1 = prob
            w1 = self.w1_0(prob)
            w1 = self.w1_0_1(w1)
            # w1 = self.w1_0_2(w1)
            w1 = self.w1_1(w1)
            w1 = self.w1_2(w1)
            w1 = self.w1_3(w1)
            # w1 = self.w1_4(w1)
            w1 = self.w1_5(w1)

            w2 = prob
            w2 = self.w2_0(prob)
            w2 = self.w2_0_1(w2)
            # w2 = self.w2_0_2(w2)
            w2 = self.w2_1(w2)
            w2 = self.w2_2(w2)
            w2 = self.w2_3(w2)
            # w2 = self.w2_4(w2)
            w2 = self.w2_5(w2)

            if self.last_layer_activation == "sigmoid":
                mu = self.concat([irs1, irs2, w1, w2]) * self.bound

            elif self.last_layer_activation == "tanh":
                mu = self.concat([irs1, irs2, w1, w2])

        else:
            prob = self.fc1(prob)
            prob = self.fc2(prob)
            prob = self.fc3(prob)
            prob = self.fc4(prob)

            if self.last_layer_activation == "sigmoid":
                mu = self.mu(prob) * self.bound

            elif self.last_layer_activation == "tanh":
                mu = self.mu(prob)

        return mu


class PowerActorNetwork(keras.Model):
    def __init__(
        self,
        fc1_dims,
        fc2_dims,
        num_of_users,
        name="PowerActor",
        chkpt_dir="../tmp/ddpg",
    ):
        super().__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, self.model_name + "_ddpg.h5"
        )

        # self.bn0 = BatchNormalization()
        self.fc1 = Dense(128, activation="relu")
        self.fc2 = Dense(num_of_users - 1, activation="sigmoid")

    def call(self, state):
        # power = self.bn0(state)
        power = self.fc1(state)
        power = self.fc2(power)
        return power


class Actor(keras.Model):
    def __init__(
        self,
        num_of_elements,
        bound,
        name="IRSActor",
        last_layer_activation="sigmoid",
        dummy_actor_input=False,
    ):
        super().__init__()
        self.model_name = name
        self.num_of_elements = num_of_elements
        self.bound = bound
        self.last_layer_activation = last_layer_activation
        self.dummy_actor_input = dummy_actor_input

        self.initilizer = GlorotNormal(23)
        self.initilizer1 = HeUniform(23)
        self.initilizer2 = HeNormal(23)

        # self.bn0 = BatchNormalization()
        # self.fc_00 = Dense(256, activation="relu")
        # self.fc_0 = Dense(750, activation="relu")
        # self.fc_1 = Dense(1024, activation="relu")
        # self.fc00 = Dense(1024, activation="relu")
        # self.fc0 = Dense(512, activation="relu")

        self.activation = PReLU()
        self.activation1 = PReLU()
        self.activation2 = PReLU()

        leaky_activation = LeakyReLU(alpha=0.4)
        leaky_activation1 = LeakyReLU(alpha=0.4)

        self.fc1 = Dense(
            1024,
            activation=self.activation,
        )

        self.fc2 = Dense(
            600,
            activation=self.activation1,
        )

        # self.fc3 = Dense(
        #     512,
        #     activation=self.activation2,
        # )

        # self.fc2 = Dense(128, activation="relu")
        # # self.fc3 = Dense(128, activation="relu")
        # self.fc4 = Dense(64, activation="relu")
        self.fc5 = Dense(
            num_of_elements,
            activation=last_layer_activation,
            kernel_initializer=self.initilizer,
        )

        # self.drop0 = Dropout(0.2)
        # self.drop1 = Dropout(0.2)
        # self.drop2 = Dropout(0.2)

    def call(self, state):
        phase = state
        if self.dummy_actor_input:
            phase = tf.ones_like(phase)
            
        # phase = self.bn0(state)

        # phase = self.fc_00(phase)
        # phase = self.fc_0(phase)

        # phase = self.drop0(phase)
        # phase = self.fc_1(phase)
        # phase = self.fc00(phase)
        # phase = self.fc0(phase)

        # phase = self.drop1(phase)

        phase = self.fc1(phase)

        # phase = self.drop2(phase)
        phase = self.fc2(phase)
        # phase = self.fc3(phase)
        # phase = self.fc4(phase)

        if self.last_layer_activation == "sigmoid":
            phase = self.fc5(phase) * self.bound

        elif self.last_layer_activation == "tanh":
            phase = self.fc5(phase)

        return phase
