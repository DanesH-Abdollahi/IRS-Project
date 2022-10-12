import os
import numpy as np
import cmath
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt


def Random_Complex_Mat(Row: int, Col: int):
    tmp = []
    for _ in range(Row):
        tmp.append(
            [cmath.exp(complex(0, random.uniform(-math.pi, math.pi)))
             for _ in range(Col)]
        )
    Matrix = np.array(tmp)
    Matrix = Matrix.reshape(Row, Col)
    return Matrix


class User:
    def __init__(
        self,
        d1: float,
        d2: float,
        d3: float,
        NoiseVar: float,
        LosToAntenna: bool,
        LosToIrs1: bool,
        LosToIrs2: bool,
    ) -> None:

        self.DistanceFromAntenna = d1
        self.DistanceFromIrs1 = d2
        self.DistanceFromIrs2 = d3
        self.NoisePower = NoiseVar

        self.LosToAntenna = LosToAntenna
        self.LosToIrs1 = LosToIrs1
        self.LosToIrs2 = LosToIrs2

        self.hsu = 0
        self.h1u = 0
        self.h2u = 0
        self.w = 0

        self.SINRThreshold = 10  # 6 dB approximately

    def GenerateMatrixes(self, env) -> None:
        if self.LosToAntenna:
            self.hsu = Random_Complex_Mat(1, env.N) / self.DistanceFromAntenna
        else:
            self.hsu = np.zeros((1, env.N))

        if self.LosToIrs1:
            self.h1u = Random_Complex_Mat(1, env.M1) / self.DistanceFromIrs1

        else:
            self.h1u = np.zeros((1, env.M1))

        if self.LosToIrs2:
            self.h2u = Random_Complex_Mat(1, env.M2) / self.DistanceFromIrs2

        else:
            self.h2u = np.zeros((1, env.M2))

        self.w = (Random_Complex_Mat(env.N, 1) / cmath.sqrt(env.N)) * 50


class Environment:
    def __init__(self) -> None:
        self.N = 5  # Number of Antennas
        self.M1 = 4  # Number of Elements of IRS1
        self.M2 = 4  # Number of Elements of IRS1

        self.PathLosExponent = 2
        self.Irs1ToAntenna = 10  # The Distance Between IRS1 & Antenna
        self.Irs2ToAntenna = 10  # The Distance Between IRS2 & Antenna
        self.Irs1ToIrs2 = 10  # The Distance Between IRS1 & IRS2

        self.Users = []
        self.SINR = []
        self.SumRate = 0

        # Generate Random Channel Coefficient Matrix(es)
        self.Hs1 = Random_Complex_Mat(self.M1, self.N) / self.Irs1ToAntenna
        self.Hs2 = Random_Complex_Mat(self.M2, self.N) / self.Irs2ToAntenna
        self.H12 = Random_Complex_Mat(self.M2, self.M1) / self.Irs1ToIrs2
        self.H21 = np.conjugate(np.transpose(self.H12))

        # Generate Initial IRS Coefficient Matrix(es)
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])

    def CreateUser(
        self,
        d1: float,
        d2: float,
        d3: float,
        NoiseVar: float,
        LosToAntenna: bool,
        LosToIrs1: bool,
        LosToIrs2: bool,
    ):
        Usr = User(d1, d2, d3, NoiseVar, LosToAntenna, LosToIrs1, LosToIrs2)
        Usr.GenerateMatrixes(self)
        self.Users.append(Usr)
        return Usr

    def Reward(self) -> float:
        self.CalculateSINR()
        reward = self.SumRate
        Penalty = 10
        for i in enumerate(self.SINR):
            if i[1] < self.Users[i[0]].SINRThreshold:
                reward -= Penalty

        return reward

    def CalculateSINR(self):
        SINR = []
        self.Psi1 = np.diag(self.state[0:4])
        self.Psi1 = np.diag(self.state[4:8])
        for u in enumerate(self.Users):
            u[1].w = np.reshape(
                self.state[8+(u[0]*self.N):8+(u[0]*self.N)+self.N], (self.N, 1))

        for i in enumerate(self.Users):
            numerator = (
                np.absolute(
                    np.dot(
                        i[1].hsu
                        + np.dot(i[1].h1u, np.dot(self.Psi1, self.Hs1))
                        + np.dot(i[1].h2u, np.dot(self.Psi2, self.Hs2))
                        + np.dot(i[1].h2u, np.dot(self.Psi2, self.Hs2))
                        + np.dot(
                            np.dot(i[1].h1u, np.dot(self.Psi1, self.H21)),
                            np.dot(self.Psi2, self.Hs2),
                        )
                        + np.dot(
                            np.dot(i[1].h2u, np.dot(self.Psi2, self.H12)),
                            np.dot(self.Psi1, self.Hs1),
                        ),
                        i[1].w,
                    )
                )
                ** 2
            )

            # print(numerator.shape)

            denominator = i[1].NoisePower
            for j in enumerate(self.Users):
                if j[0] != i[0]:
                    denominator += (
                        np.absolute(
                            np.dot(
                                j[1].hsu
                                + np.dot(j[1].h1u, np.dot(self.Psi1, self.Hs1))
                                + np.dot(j[1].h2u, np.dot(self.Psi2, self.Hs2))
                                + np.dot(j[1].h2u, np.dot(self.Psi2, self.Hs2))
                                + np.dot(
                                    np.dot(j[1].h1u, np.dot(
                                        self.Psi1, self.H21)),
                                    np.dot(self.Psi2, self.Hs2),
                                )
                                + np.dot(
                                    np.dot(j[1].h2u, np.dot(
                                        self.Psi2, self.H12)),
                                    np.dot(self.Psi1, self.Hs1),
                                ),
                                j[1].w,
                            )
                        )
                        ** 2
                    )
            SINR.append((numerator / denominator)[0, 0])

        self.SINR = SINR
        self.SumRate = sum(math.log2(1 + i) for i in self.SINR)

    def State(self) -> np.ndarray:
        self.state = np.concatenate(
            (
                np.diag(self.Psi1),  # M1
                np.diag(self.Psi2),  # M2
                self.Users[0].w[:, 0],  # N
                [0, 0]
            ),
            axis=0,
        )
        self.CalculateSINR()
        self.state = np.concatenate(
            (
                np.diag(self.Psi1),  # M1
                np.diag(self.Psi2),  # M2
                self.Users[0].w[:, 0],  # N
                self.SINR,  # Users Num.
                [self.SumRate],  # 1
            ),
            axis=0,
        )
        return self.state

    def reset(self):
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])
        for i in enumerate(self.Users):
            i[1].w = (Random_Complex_Mat(self.N, 1) / cmath.sqrt(self.N)) * 50

        return self.State()

    def step(self, action):
        self.state = self.state + np.concatenate(
            (
                action[0:self.M1],  # M1
                action[self.M1:self.M1 + self.M2],  # M2
                action[self.M1 + self.M2: self.M1 + self.M2 + self.N],  # N
                [0, 0]
            ),
            axis=0,
        )

        return self.state, self.Reward(), False, self.SumRate

        # def SumRate(self):
        #     return sum(math.log2(1 + i) for i in self.SINR)

        # class Agent:
        #     def __init__(self, env: Environment):
        #         self.Env = env
        #         self.QFunction = dict()
        #         self.State = env.State()
        #         self.Action = np.zeros((env.N, 1))
        #         self.Reward = 0
        #         self.NextState = np.zeros(
        #             (env.N + env.M1 + env.M2 + len(env.Users) + 1, 1))
        #         self.NextAction = np.zeros((env.N, 1))
        #         self.Episode = 0
        #         self.Step = 0
        #         self.Epsilon = 0.1
        #         self.Alpha = 0.1
        #         self.Gamma = 0.9
        #         self.EpsilonDecay = 0.999
        #         self.AlphaDecay = 0.999
        #         self.GammaDecay = 0.999

        #     def TakeAction(self):
        #         if np.random.rand() < self.Epsilon:
        #             self.Action = np.random.rand(self.Env.N, 1)
        #         else:
        #             self.Action = self.QFunction.get(
        #                 tuple(self.State), np.random.rand(self.Env.N, 1)
        #             )

        #         self.Env.Users[0].w = self.Action
        #         self.Reward = self.Env.Reward()
        #         self.NextState = self.Env.State()


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None) -> None:
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) *
            np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else \
            np.zeros_like(self.mu)


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)

        self.new_state_memory = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)

        dtype = np.int8 if n_actions < 256 else np.int16
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)  # if done is True, then 0
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir="tmp/ddpg"):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.sess = sess
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir

        self.build_network()

        self.params = tf.compat.v1.trainable_variables(scope=self.name)
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, name + "_ddpg.ckpt")

        self.unnormalized_actor_gradients = tf.compat.v1.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.compat.v1.div(
            x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).apply_gradients(
            zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(
                tf.compat.v1.float32, shape=[None, *self.input_dims], name="inputs"
            )
            self.action_gradient = tf.compat.v1.placeholder(
                tf.compat.v1.float32, shape=[None, self.n_actions])

            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.compat.v1.layers.dense(inputs=self.input, units=self.fc1_dims,
                                               kernel_initializer=tf.compat.v1.random_uniform_initializer(
                                                   minval=-f1, maxval=f1),
                                               bias_initializer=tf.compat.v1.random_uniform_initializer(minval=-f1, maxval=f1))
            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.compat.v1.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=self.fc2_dims,
                                               kernel_initializer=tf.compat.v1.random_uniform_initializer(
                                                   minval=-f2, maxval=f2),
                                               bias_initializer=tf.compat.v1.random_uniform_initializer(minval=-f2, maxval=f2))
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.compat.v1.nn.relu(batch2)

            f3 = 0.003
            mu = tf.compat.v1.layers.dense(
                layer2_activation,
                units=self.n_actions,
                activation='tanh',
                kernel_initializer=tf.compat.v1.random_uniform_initializer(
                    minval=-f3, maxval=f3),
                bias_initializer=tf.compat.v1.random_uniform_initializer(
                    minval=-f3, maxval=f3),
            )

            self.mu = tf.compat.v1.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={
                      self.input: inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.saver.restore(self.sess, self.checkpoint_file)


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, batch_size=64, chkpt_dir="tmp/ddpg"):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.sess = sess
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir

        self.build_network()
        self.params = tf.compat.v1.trainable_variables(scope=self.name)
        self.saver = tf.compat.v1.train.Saver()
        self.checkpoint_file = os.path.join(
            self.chkpt_dir, name + "_ddpg.ckpt")

        self.optimize = tf.compat.v1.train.AdamOptimizer(
            self.lr).minimize(self.loss)

        self.action_gradients = tf.compat.v1.gradients(self.q, self.actions)

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.compat.v1.placeholder(
                tf.compat.v1.float32, shape=[None, *self.input_dims], name="inputs")

            self.actions = tf.compat.v1.placeholder(
                tf.compat.v1.float32, shape=[None, self.n_actions], name="actions"
            )

            self.q_target = tf.compat.v1.placeholder(
                tf.compat.v1.float32, shape=[None, 1], name="targets")

            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.compat.v1.layers.dense(inputs=self.input, units=self.fc1_dims,
                                               kernel_initializer=tf.compat.v1.random_uniform_initializer(
                                                   minval=-f1, maxval=f1),
                                               bias_initializer=tf.compat.v1.random_uniform_initializer(minval=-f1, maxval=f1))

            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.compat.v1.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=self.fc2_dims,
                                               kernel_initializer=tf.compat.v1.random_uniform_initializer(
                                                   minval=-f2, maxval=f2),
                                               bias_initializer=tf.compat.v1.random_uniform_initializer(minval=-f2, maxval=f2))
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)

            action_in = tf.compat.v1.layers.dense(
                self.actions, units=self.fc2_dims, activation='relu')

            state_actions = tf.compat.v1.add(batch2, action_in)
            state_actions = tf.compat.v1.nn.relu(state_actions)

            f3 = 0.003
            self.q = tf.compat.v1.layers.dense(state_actions, units=1, kernel_initializer=tf.compat.v1.random_uniform_initializer(
                minval=-f3, maxval=f3), bias_initializer=tf.compat.v1.random_uniform_initializer(minval=-f3, maxval=f3),
                kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01))

            self.loss = tf.compat.v1.losses.mean_squared_error(
                self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.actions: actions, self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients, feed_dict={self.input: inputs, self.actions: actions})

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.saver.restore(self.sess, self.checkpoint_file)


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        # self.sess = tf.Session() ?????
        self.sess = tf.compat.v1.Session()

        self.actor = Actor(alpha, n_actions, "Actor", input_dims,
                           self.sess, layer1_size, layer2_size, math.pi)

        self.critic = Critic(beta, n_actions, "Critic",
                             input_dims, self.sess, layer1_size, layer2_size)

        self.target_actor = Actor(alpha, n_actions, "TargetActor",
                                  input_dims, self.sess, layer1_size, layer2_size, math.pi)

        self.target_critic = Critic(beta, n_actions, "TargetCritic",
                                    input_dims, self.sess, layer1_size, layer2_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = [self.target_critic.params[i].assign(tf.compat.v1.multiply(self.critic.params[i], self.tau) +
                                                                  tf.compat.v1.multiply(self.target_critic.params[i], 1. - self.tau))
                              for i in range(len(self.target_critic.params))]

        self.update_actor = [self.target_actor.params[i].assign(tf.compat.v1.multiply(self.actor.params[i], self.tau) +
                                                                tf.compat.v1.multiply(self.target_actor.params[i], 1. - self.tau))
                             for i in range(len(self.target_actor.params))]

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau

        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def choose_action(self, state):
        state = state[np.newaxis, :]  # add a dimension ?
        mu = self.actor.predict(state)
        mu_prime = mu + self.noise()
        return mu_prime[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)

        target_actions = self.target_actor.predict(new_state)
        critic_value_ = self.target_critic.predict(new_state, target_actions)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])

        target = np.reshape(target, (self.batch_size, 1))
        _ = self.critic.train(state, action, target)
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


def Run():
    tf.compat.v1.disable_eager_execution()
    env = Environment()
    # U1 = env.CreateUser(17.5, 10, 10, 1, False, True, False)
    U2 = env.CreateUser(25, 15, 15, 1, True, True, True)

    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[15], tau=0.001,
                  env=env, batch_size=64, layer1_size=400, layer2_size=300, n_actions=13)

    # print(abs(env.State()))

    np.random.seed(23)
    score_history = []

    for iter in range(1):
        obs = env.reset()
        done = False
        score = 0
        rewards = np.zeros((1, 1000))
        sumrate = np.zeros((1, 1000))
        # U1_SINR = np.zeros((1, 1000))
        # U2_SINR = np.zeros((1, 1000))

        for i in range(1000):
            action = agent.choose_action(obs)
            new_state, reward, done, sumrate[iter][i] = env.step(action)
            agent.remember(obs, action, reward, new_state, done)
            agent.learn()
            score += reward
            rewards[iter][i] = reward
            obs = new_state
            # U1_SINR[iter][i] = SINRs[0]
            # U2_SINR[iter][i] = SINRs[1]

        score_history.append(score)
        # print('Episode', iter + 1, " | ", 'Score -> %.2f' % score, " | ",
        #       'Avg of last 20 episodes -> %.3f' % np.mean(score_history[-20:]))

        print(
            f"{'Episode'} {iter + 1: < 4} {' | '} {'Score -> '} {score: < 10.2f} {' | '}{'Avg_Score of last 20 episodes ->'}{np.mean(score_history[-20:]): < 10.2f}")

    # plt.plot(range(1, len(score_history)+1), score_history)
    # plt.ylabel('Score')
    # plt.xlabel('Episode')
    # plt.grid(1)
    # plt.savefig('tmp_results/Score.png')
    # plt.show()

    rewards = np.mean(rewards, axis=0)
    plt.plot(range(1, len(rewards)+1), rewards)
    plt.ylabel('Mean Rewards')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.savefig('tmp_results/Mean_Rewards.png')
    plt.show()

    sumrate = np.mean(sumrate, axis=0)
    plt.plot(range(1, len(sumrate)+1), sumrate)
    plt.ylabel('Mean Sumrate')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.savefig('tmp_results/Mean_Sumrate.png')
    plt.show()

    # plt.plot(range(1, len(U1_SINR)+1), U1_SINR)
    # plt.ylabel('U1 SINR')
    # plt.xlabel('Iteration')
    # plt.grid(1)
    # plt.savefig('tmp_results/U1_SINR.png')
    # plt.show()

    # plt.plot(range(1, len(U2_SINR)+1), U2_SINR)
    # plt.ylabel('U2 SINR')
    # plt.xlabel('Iteration')
    # plt.grid(1)
    # plt.savefig('tmp_results/U2_SINR.png')
    # plt.show()


if __name__ == "__main__":
    Run()
