import numpy as np
from math import pi
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Buffer import ReplayBuffer
from Networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.001, env=None,
                 gamma=0.99, n_actions=2, max_size=100000, tau=0.001,
                 fc1=256, fc2=128, batch_size=32, noise=0.0025):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = 1
        self.min_action = -1

        self.actor = ActorNetwork(
            n_actions=self.n_actions, name='Actor', fc1_dims=fc1, fc2_dims=fc2)
        self.critic = CriticNetwork(name='Critic', fc1_dims=fc1, fc2_dims=fc2)

        self.target_actor = ActorNetwork(
            n_actions=self.n_actions, name='TargetActor', fc1_dims=fc1, fc2_dims=fc2)

        self.target_critic = CriticNetwork(
            name='TargetCritic', fc1_dims=fc1, fc2_dims=fc2)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(
            learning_rate=alpha))  # needed ?
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)  # Hard update

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)

        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0,
                                        stddev=self.noise)

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state = self.memory.sample_buffer(
            self.batch_size)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        new_satate = tf.convert_to_tensor(new_state, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_satate)
            critic_value_ = tf.squeeze(
                self.target_critic(new_satate, target_actions), 1)
            critic_value = tf.squeeze(self.critic(state, action), 1)

            target = reward + self.gamma * critic_value_
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(
            critic_loss, self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(
            zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(state)
            actor_loss = -self.critic(state, new_policy_actions)

            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(
            actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
