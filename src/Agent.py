import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from Buffer import Buffer
from Networks import ActorNetwork, CriticNetwork, PowerActorNetwork


class Agent:
    def __init__(
        self,
        num_states,
        n_actions,
        bound,
        alpha=0.001,
        beta=0.002,
        env=None,
        gamma=0.99,
        max_size=100000,
        tau=0.005,
        fc1=512,
        fc2=256,
        batch_size=256,
        noise=0.055,
        warmup=0,
        uniform_selection=True,
        TD3=False,
        TD3_update_interval=2,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(
            num_states, n_actions, buffer_capacity=max_size, batch_size=batch_size
        )
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.bounds = bound
        self.max_action = bound
        self.min_action = -bound
        self.env = env

        self.power_noise = 0
        self.step = 0
        self.warmup = warmup
        self.uniform_selection = uniform_selection
        self.TD3 = TD3
        self.update_interval = TD3_update_interval

        self.actor = ActorNetwork(
            fc1_dims=fc1,
            fc2_dims=fc2,
            bound=self.bounds,
            n_actions=self.n_actions,
            name="Actor",
        )

        self.target_actor = ActorNetwork(
            fc1_dims=fc1,
            fc2_dims=fc2,
            bound=self.bounds,
            n_actions=self.n_actions,
            name="TargetActor",
        )

        if self.TD3:
            self.critic_1 = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name="Critic_1")
            self.critic_2 = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name="Critic_2")
            self.target_critic_1 = CriticNetwork(
                fc1_dims=fc1, fc2_dims=fc2, name="TargetCritic_1"
            )
            self.target_critic_2 = CriticNetwork(
                fc1_dims=fc1, fc2_dims=fc2, name="TargetCritic_2"
            )

        else:
            self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name="Critic")

            self.target_critic = CriticNetwork(
                fc1_dims=fc1, fc2_dims=fc2, name="TargetCritic"
            )

        self.power = PowerActorNetwork(
            fc1_dims=128, fc2_dims=32, num_of_users=env.num_of_users, name="PowerActor"
        )
        self.target_power = PowerActorNetwork(
            fc1_dims=128, fc2_dims=32, num_of_users=env.num_of_users, name="TargetPower"
        )

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))

        if self.TD3:
            self.critic_1.compile(optimizer=Adam(learning_rate=beta))
            self.critic_2.compile(optimizer=Adam(learning_rate=beta))
            self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
            self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        else:
            self.critic.compile(optimizer=Adam(learning_rate=beta))
            self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.power.compile(optimizer=Adam(learning_rate=alpha / 2))
        self.target_power.compile(optimizer=Adam(learning_rate=beta / 2))

        self.update_network_parameters(tau=1)  # Hard update

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        if self.TD3:
            if self.step % self.update_interval != 0:
                return
            
            
        for a, b in zip(self.target_actor.weights, self.actor.weights):
            a.assign(b * tau + a * (1 - tau))

        if self.TD3:
            for a, b in zip(self.target_critic_1.weights, self.critic_1.weights):
                a.assign(b * tau + a * (1 - tau))

            for a, b in zip(self.target_critic_2.weights, self.critic_2.weights):
                a.assign(b * tau + a * (1 - tau))

        else:
            for a, b in zip(self.target_critic.weights, self.critic.weights):
                a.assign(b * tau + a * (1 - tau))

        for a, b in zip(self.target_power.weights, self.power.weights):
            a.assign(b * tau + a * (1 - tau))

    def remember(self, state, action, reward, new_state):
        self.memory.record((state, action, reward, new_state))

    def save_models(self):
        print("... saving models ...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        power_action = self.power(state)

        if not evaluate:
            action_noise = tf.random.normal(
                shape=[self.n_actions - 1], mean=0, stddev=self.noise
            )
            actions += action_noise

            self.power_noise = tf.random.normal(
                shape=[self.env.num_of_users - 1], mean=0, stddev=self.noise / 2
            )
            power_action += self.power_noise

            # noise = tf.concat([action_noise, power_noise], axis=0)
            # actions += action_noise

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        power_action = tf.clip_by_value(power_action, 0, 1)
        actions = tf.concat([actions, power_action], axis=1)

        return actions[0]

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape(persistent=True) as tape:
            if self.TD3:
                target_actions = self.target_actor(next_state_batch)
                target_power_action = self.target_power(next_state_batch)

                target_action_noise = tf.random.normal(
                    shape=[self.batch_size, self.n_actions - 1],
                    mean=0,
                    stddev=self.noise,
                )

                target_action_noise = tf.clip_by_value(target_action_noise, -0.2, 0.2)

                target_actions += target_action_noise

                target_power_noise = tf.random.normal(
                    shape=[self.batch_size, self.env.num_of_users - 1],
                    mean=0,
                    stddev=self.noise / 2,
                )
                target_power_noise = tf.clip_by_value(target_power_noise, -0.1, 0.1)

                target_power_action += target_power_noise
                target_power_action = tf.clip_by_value(target_power_action, 0, 1)

                target_actions = tf.concat(
                    [target_actions, target_power_action], axis=1
                )

                target_actions = tf.clip_by_value(
                    target_actions, self.min_action, self.max_action
                )

            else:
                target_actions = self.target_actor(next_state_batch)
                target_power_action = self.target_power(next_state_batch)
                target_actions = tf.concat(
                    [target_actions, target_power_action], axis=1
                )

            if self.TD3:
                y = reward_batch + self.gamma * tf.math.minimum(
                    self.target_critic_1(next_state_batch, target_actions),
                    self.target_critic_2(next_state_batch, target_actions),
                )
                critic_value_1 = self.critic_1(state_batch, action_batch)
                critic_value_2 = self.critic_2(state_batch, action_batch)
                critic_loss_1 = tf.math.reduce_mean(tf.math.square(y - critic_value_1))
                critic_loss_2 = tf.math.reduce_mean(tf.math.square(y - critic_value_2))

            else:
                y = reward_batch + self.gamma * self.target_critic(
                    next_state_batch, target_actions
                )
                critic_value = self.critic(state_batch, action_batch)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        if self.TD3:
            critic_grad_1 = tape.gradient(
                critic_loss_1, self.critic_1.trainable_variables
            )
            self.critic_1.optimizer.apply_gradients(
                zip(critic_grad_1, self.critic_1.trainable_variables)
            )

            critic_grad_2 = tape.gradient(
                critic_loss_2, self.critic_2.trainable_variables
            )
            self.critic_2.optimizer.apply_gradients(
                zip(critic_grad_2, self.critic_2.trainable_variables)
            )

        else:
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )

        if self.TD3:
            if self.step % self.update_interval != 0:
                return

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            power_action = self.power(state_batch)
            actions = tf.concat([actions, power_action], axis=1)

            if self.TD3:
                actor_loss = -tf.math.reduce_mean(self.critic_1(state_batch, actions))
            else:
                critic_value = self.critic(state_batch, actions)
                actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            power_action = self.power(state_batch)
            actions = tf.concat([actions, power_action], axis=1)

            if self.TD3:
                actor_loss = -tf.math.reduce_mean(self.critic_1(state_batch, actions))
            else:
                actor_loss = -tf.math.reduce_mean(self.critic(state_batch, actions))

        power_grad = tape.gradient(actor_loss, self.power.trainable_variables)
        self.power.optimizer.apply_gradients(
            zip(power_grad, self.power.trainable_variables)
        )

    def learn(self):
        self.step += 1
        if self.step < self.warmup:
            return

        if self.memory.buffer_counter < self.batch_size:
            return

        record_range = min(self.memory.buffer_counter, self.memory.buffer_capacity)

        if not self.uniform_selection:
            rewards = np.squeeze(self.memory.reward_buffer)[:record_range]
            min_reward = np.min(rewards)
            if min_reward < 0:
                rewards += abs(min_reward)

            sum_reward = np.sum(rewards)
            p = rewards / sum_reward

            # print(record_range, len(p))
            batch_indices = np.random.choice(
                record_range, self.batch_size, replace=True, p=p
            )

        batch_indices = np.random.choice(record_range, self.batch_size, replace=False)

        state_batch = tf.convert_to_tensor(self.memory.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.memory.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.memory.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.memory.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
        self.update_network_parameters()
