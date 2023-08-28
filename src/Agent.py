import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from Buffer import Buffer
from Networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, n_actions, alpha=0.0003,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions)
        self.critic = CriticNetwork()
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))

        self.memory = Buffer(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation])
        # state = observation
        probs = self.actor(state)
        probs = probs.numpy()

        # print(probs)

        # print(f"n actions: {self.n_actions}")
        dist = tfp.distributions.Normal(probs, 1.0)
        # dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        action = dist.sample()

        log_probs = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_probs = log_probs.numpy()[0]

        return action, log_probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * \
                        (reward_arr[k] + self.gamma * values[k + 1]
                         * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])

                    dist = tfp.distributions.Normal(self.actor(states), 1.0)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value)

                    prob_ratio = tf.math.exp(new_probs - old_probs)

                    # print(len(advantage[batch]))
                    weighted_probs = np.zeros(
                        (len(advantage[batch]), self.n_actions))

                    for i in range(len(advantage[batch])):
                        weighted_probs[i:] = advantage[batch][i] * \
                            prob_ratio[i].numpy()

                    # weighted_probs = advantage[batch] * prob_ratio.numpy()
                    weighted_probs = tf.convert_to_tensor(weighted_probs)

                    clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip)

                    weighted_clipped_probs = np.zeros(
                        (len(advantage[batch]), self.n_actions))

                    for i in range(len(advantage[batch])):
                        weighted_clipped_probs[i:] = advantage[batch][i] * \
                            clipped_probs[i].numpy()

                    weighted_clipped_probs = tf.convert_to_tensor(
                        weighted_clipped_probs)

                    # weighted_clipped_probs = clipped_probs * advantage[batch]

                    actor_loss = - \
                        tf.math.reduce_mean(tf.math.minimum(
                            weighted_probs, weighted_clipped_probs))

                    returns = advantage[batch] + values[batch]

                    critic_loss = tf.math.reduce_mean(
                        (returns - critic_value) ** 2)

                actor_prams = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables

                actor_grads = tape.gradient(actor_loss, actor_prams)
                critic_grads = tape.gradient(critic_loss, critic_params)

                self.actor.optimizer.apply_gradients(
                    zip(actor_grads, actor_prams))
                self.critic.optimizer.apply_gradients(
                    zip(critic_grads, critic_params))

        self.memory.clear_memory()
