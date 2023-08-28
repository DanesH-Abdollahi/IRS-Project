import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Buffer import Buffer
from Networks import ActorNetwork, CriticNetwork, PowerActorNetwork
from tensorflow import keras


class Agent:
    def __init__(self, num_states, n_actions, bound, alpha=0.001, beta=0.002,
                 env=None, gamma=0.99, max_size=100000, tau=0.005,
                 fc1=512, fc2=256, batch_size=128, noise=0.055, warmup=1000,
                 interval_update=2):

        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(num_states, n_actions,
                             buffer_capacity=max_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.bounds = bound
        self.max_action = bound
        self.min_action = -bound
        self.env = env
        self.warmup = warmup
        self.interval_update = interval_update

        self.time_step = 0
        self.learn_step_counter = 0
        self.power_noise = 0

        self.actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, bound=self.bounds,
                                  n_actions=self.n_actions, name='Actor')

        self.target_actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, bound=self.bounds,
                                         n_actions=self.n_actions, name='TargetActor')

        self.critic_1 = CriticNetwork(
            fc1_dims=fc1, fc2_dims=fc2, name='Critic_1')
        self.critic_2 = CriticNetwork(
            fc1_dims=fc1, fc2_dims=fc2, name='Critic_2')

        self.target_critic_1 = CriticNetwork(
            fc1_dims=fc1, fc2_dims=fc2, name='TargetCritic_1')

        self.target_critic_2 = CriticNetwork(
            fc1_dims=fc1, fc2_dims=fc2, name='TargetCritic_2')

        self.power = PowerActorNetwork(
            fc1_dims=128, fc2_dims=32, num_of_users=env.num_of_users, name='PowerActor')
        self.target_power = PowerActorNetwork(
            fc1_dims=128, fc2_dims=32, num_of_users=env.num_of_users, name='TargetPower')

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss="mean")
        self.critic_1.compile(optimizer=Adam(
            learning_rate=beta), loss="mean squared error")
        self.critic_2.compile(optimizer=Adam(
            learning_rate=beta), loss="mean squared error")
        self.target_actor.compile(optimizer=Adam(
            learning_rate=alpha), loss="mean")
        self.target_critic_1.compile(optimizer=Adam(
            learning_rate=beta), loss="mean squared error")
        self.target_critic_2.compile(optimizer=Adam(
            learning_rate=beta), loss="mean squared error")

        self.power.compile(optimizer=Adam(learning_rate=alpha))
        self.target_power.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)  # Hard update

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for (a, b) in zip(self.target_actor.weights, self.actor.weights):
            a.assign(b * tau + a * (1 - tau))

        for (a, b) in zip(self.target_critic_1.weights, self.critic_1.weights):
            a.assign(b * tau + a * (1 - tau))

        for (a, b) in zip(self.target_critic_2.weights, self.critic_2.weights):
            a.assign(b * tau + a * (1 - tau))

        for (a, b) in zip(self.target_power.weights, self.power.weights):
            a.assign(b * tau + a * (1 - tau))

    def remember(self, state, action, reward, new_state):
        self.memory.store((state, action, reward, new_state))

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

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            actions = np.random.normal(
                scale=self.noise, size=(self.n_actions-1))
            power_action = np.random.normal(
                scale=self.noise/2, size=(self.env.num_of_users-1))

            actions = np.clip(actions, self.min_action, self.max_action)
            power_action = np.clip(power_action, 0, 1)
            actions = np.concatenate([actions, power_action], axis=0)

            actions = np.float32(actions)

            return actions

        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.actor(state)
            power_action = self.power(state)

        action_noise = tf.random.normal(shape=[self.n_actions-1], mean=0,
                                        stddev=self.noise)
        actions += action_noise

        self.power_noise = tf.random.normal(shape=[self.env.num_of_users-1], mean=0,
                                            stddev=self.noise/2)

        self.power_noise = tf.clip_by_value(self.power_noise, -0.4, 0.4)
        power_action += self.power_noise

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        power_action = tf.clip_by_value(power_action, 0, 1)
        actions = tf.concat([actions, power_action], axis=1)

        return actions[0]

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_state_batch) + tf.clip_by_value(
                tf.random.normal(shape=[self.n_actions-1], mean=0, stddev=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(
                target_actions, self.min_action, self.max_action)

            target_power_action = self.target_power(next_state_batch) + tf.clip_by_value(
                tf.random.normal(shape=[self.env.num_of_users-1], mean=0, stddev=0.2), -0.2, 0.2)

            target_power_action = tf.clip_by_value(target_power_action, 0, 1)

            target_actions = tf.concat(
                [target_actions, target_power_action], axis=1)

            q1_ = self.target_critic_1(next_state_batch, target_actions)
            q2_ = self.target_critic_2(next_state_batch, target_actions)

            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            q1 = self.critic_1(state_batch, action_batch)
            q2 = self.critic_2(state_batch, action_batch)

            q1 = tf.squeeze(q1, 1)
            q2 = tf.squeeze(q2, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            target = reward_batch + self.gamma * critic_value_

            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_grad = tape.gradient(
            critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grad = tape.gradient(
            critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(
            zip(critic_1_grad, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(
            zip(critic_2_grad, self.critic_2.trainable_variables))

        self.learn_step_counter += 1

        if self.learn_step_counter % self.interval_update != 0:
            return

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            power_action = self.power(state_batch)
            actions = tf.concat([actions, power_action], axis=1)

            critic_1_value = self.critic_1(state_batch, actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_grad = tape.gradient(
            actor_loss, self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            power_action = self.power(state_batch)
            actions = tf.concat([actions, power_action], axis=1)

            actor_loss = - \
                tf.math.reduce_mean(self.critic_1(state_batch, actions))

        power_grad = tape.gradient(
            actor_loss, self.power.trainable_variables)
        self.power.optimizer.apply_gradients(
            zip(power_grad, self.power.trainable_variables))

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        states, actions, rewards, next_states = self.memory.sample_buffer()

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)

        self.update(states, actions, rewards, next_states)
        self.update_network_parameters()
