import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=1000000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros(
            (self.buffer_capacity, num_states), dtype=np.float32)
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, num_states), dtype=np.float32)
        self.action_buffer = np.zeros(
            (self.buffer_capacity, num_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity), dtype=np.float32)

    def store(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample_buffer(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        states = self.state_buffer[batch_indices]
        actions = self.action_buffer[batch_indices]
        rewards = self.reward_buffer[batch_indices]
        next_states = self.next_state_buffer[batch_indices]

        return states, actions, rewards, next_states
