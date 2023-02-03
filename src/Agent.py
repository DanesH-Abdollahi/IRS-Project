import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Buffer import Buffer
from Networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, num_states, n_actions, bound, alpha=0.001, beta=0.002,
                 env=None, gamma=0.99, max_size=100000, tau=0.005,
                 fc1=256, fc2=128, batch_size=64, noise=0.0025):
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(num_states, n_actions,
                             buffer_capacity=max_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.bounds = bound
        self.max_action = bound
        self.min_action = -1 * bound

        self.actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, bound=self.bounds,
                                  n_actions=self.n_actions, name='Actor')

        self.target_actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, bound=self.bounds,
                                         n_actions=self.n_actions, name='TargetActor')

        self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='Critic')

        self.target_critic = CriticNetwork(
            fc1_dims=fc1, fc2_dims=fc2, name='TargetCritic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)  # Hard update

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for (a, b) in zip(self.target_actor.weights, self.actor.weights):
            a.assign(b * tau + a * (1 - tau))

        for (a, b) in zip(self.target_critic.weights, self.critic.weights):
            a.assign(b * tau + a * (1 - tau))

    def remember(self, state, action, reward, new_state):
        self.memory.record((state, action, reward, new_state))

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

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * \
                self.target_critic(next_state_batch, target_actions)
            critic_value = self.critic(state_batch, action_batch)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic(state_batch, actions)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        record_range = min(self.memory.buffer_counter,
                           self.memory.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(
            self.memory.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(
            self.memory.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(
            self.memory.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.memory.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
        self.update_network_parameters()
