from Functions import *
from User import User
from cmath import sqrt
from math import log2, log10
from math import prod


class Environment:
    def __init__(
        self,
        *,
        num_of_antennas: int,
        num_of_irs1: int,
        num_of_irs2: int,
        path_loss_exponent: float,
        irs1_to_antenna: float,
        irs2_to_antenna: float,
        irs1_to_irs2: float,
        transmitted_power: float,
        reward_function: str,
        state_dB: bool,
        without_irs: bool,
    ) -> None:
        self.N = num_of_antennas  # Number of Antennas
        self.M1 = num_of_irs1  # Number of Elements of IRS1
        self.M2 = num_of_irs2  # Number of Elements of IRS2

        self.path_loss_exponent = path_loss_exponent
        self.irs1_to_antenna = irs1_to_antenna  # The Distance Between IRS1 & Antenna
        self.irs2_to_antenna = irs2_to_antenna  # The Distance Between IRS2 & Antenna
        self.irs1_to_irs2 = irs1_to_irs2  # The Distance Between IRS1 & IRS2
        self.transmitted_power = transmitted_power  # The Power Transmitted to Antenna
        self.Users = []
        self.SINR = []
        self.SumRate = 0
        self.num_of_users = 0
        self.reward_function = reward_function
        self.state_dB = state_dB
        # self.power_split_factor = 0.5
        # self.power_factors = [0.5, 0.5]
        self.sumrates_array = np.zeros((10000,))
        self.iter = 0
        self.max_sumrate = 0
        self.without_irs = without_irs

        # Generate Random Channel Coefficient Matrix(es)
        self.Hs1 = Random_Complex_Mat(self.M1, self.N) / self.irs1_to_antenna
        self.Hs2 = Random_Complex_Mat(self.M2, self.N) / self.irs2_to_antenna
        self.H12 = Random_Complex_Mat(self.M2, self.M1) / self.irs1_to_irs2
        self.H21 = np.conjugate(np.transpose(self.H12))

        # Generate Initial IRS Coefficient Matrix(es)
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])
        # self.W = np.zeros((self.N, 2), dtype=complex)

    def CreateUser(
        self,
        *,
        distance_to_antenna: float,
        distance_to_irs1: float,
        distance_to_irs2: float,
        noise_var: float,
        los_to_antenna: bool,
        los_to_irs1: bool,
        los_to_irs2: bool,
        sinr_threshold: float,
        penalty: float,
        allocated_power: float,
        weight: float,
    ):
        Usr = User(
            distance_to_antenna,
            distance_to_irs1,
            distance_to_irs2,
            noise_var,
            los_to_antenna,
            los_to_irs1,
            los_to_irs2,
            sinr_threshold,
            penalty,
            allocated_power,
            weight,
        )

        Usr.GenerateMatrixes(self)
        self.Users.append(Usr)
        self.SINR.append(0)
        self.num_of_users += 1
        # self.W[:, self.num_of_users - 1] = Usr.w[:, 0]
        return Usr

    def CalculateSINR(self):
        SINR = []
        for i in enumerate(self.Users):
            numerator = (
                np.abs(
                    (
                        i[1].hsu
                        + (i[1].h1u @ self.Psi1) @ self.Hs1
                        + (i[1].h2u @ self.Psi2) @ self.Hs2
                        # + (i[1].h2u @ self.Psi2) @ self.Hs2
                        + ((i[1].h1u @ self.Psi1) @ self.H21) @ (self.Psi2 @ self.Hs2)
                        + ((i[1].h2u @ self.Psi2) @ self.H12) @ (self.Psi1 @ self.Hs1)
                    )
                    @ i[1].w
                )
                ** 2
            )

            denominator = i[1].noise_power
            for j in enumerate(self.Users):
                if j[0] != i[0]:
                    denominator += (
                        np.abs(
                            (
                                i[1].hsu
                                + (i[1].h1u @ self.Psi1) @ self.Hs1
                                + (i[1].h2u @ self.Psi2) @ self.Hs2
                                # + (j[1].h2u @ self.Psi2) @ self.Hs2
                                + ((i[1].h1u @ self.Psi1) @ self.H21)
                                @ (self.Psi2 @ self.Hs2)
                                + ((i[1].h2u @ self.Psi2) @ self.H12)
                                @ (self.Psi1 @ self.Hs1)
                            )
                            @ j[1].w
                        )
                        ** 2
                    )

            SINR.append((numerator / denominator)[0, 0])

        # self.SINR = [10*log10(i) for i in SINR]
        self.SINR = SINR
        self.SumRate = sum(log2(1 + ii) for ii in SINR)

    def Reward(self) -> float:
        weighted_reward = sum(
            self.Users[i[0]].weight * log2(1 + i[1]) for i in enumerate(self.SINR)
        )

        product_rate = prod(log2(1 + i) for i in self.SINR)

        if self.reward_function == "product*sumrate":
            reward = product_rate * weighted_reward

        elif self.reward_function == "sumrate":  # Bad
            reward = weighted_reward

        elif self.reward_function == "product":
            reward = product_rate

        elif self.reward_function == "product+sumrate":  # Bad
            reward = product_rate + weighted_reward

        elif self.reward_function == "sumrate_with_penalty":  # Not Good
            reward = weighted_reward
            for i in enumerate(self.SINR):
                if i[1] < self.Users[i[0]].sinr_threshold:
                    reward -= self.Users[i[0]].penalty

        elif self.reward_function == "man":
            reward = (weighted_reward**2) * product_rate

        elif self.reward_function == "man2":
            reward = (weighted_reward**3) * product_rate

        elif self.reward_function == "man3":  # Usually Better than man2
            reward = (weighted_reward**3) * (product_rate ** (1 / 3))

        elif self.reward_function == "man4":  # Good
            reward = (weighted_reward**2) * (product_rate ** (1 / 3))

        elif self.reward_function == "man5":  # Good
            reward = (weighted_reward**4) * (product_rate ** (1 / 3))

        # --------------------------------------------------------------------------------

        elif self.reward_function == "man6":  # Good
            reward = 0.65 * (weighted_reward**3) * product_rate + (
                0.35 * weighted_reward
            )

        elif self.reward_function == "man6_2":
            reward = 0.75 * (weighted_reward**3) * product_rate + (
                0.25 * weighted_reward
            )

        elif self.reward_function == "man6_3":
            reward = 0.65 * (weighted_reward**4) * product_rate + (
                0.35 * weighted_reward
            )

        elif self.reward_function == "man6_4":
            reward = 0.8 * (weighted_reward**3) * product_rate + (
                0.2 * (weighted_reward**1.5)
            )

        # --------------------------------------------------------------------------------

        elif self.reward_function == "man7":
            reward = 0.65 * (weighted_reward**3) * (product_rate ** (1 / 3)) + (
                0.35 * weighted_reward
            )

        elif self.reward_function == "man8":
            if product_rate >= 1:
                reward = (
                    0.65 * (weighted_reward**2) * (product_rate ** (1 / 2))
                    + 0.35 * weighted_reward
                )

            else:
                reward = (
                    0.65 * (weighted_reward**2) * (product_rate ** (2))
                    + 0.35 * weighted_reward
                )

        elif self.reward_function == "man9":
            reward = (weighted_reward**2) * (product_rate ** (1 / 2))

        return reward

    def State(self) -> None:
        self.CalculateSINR()

        self.sumrates_array[self.iter] = self.SumRate
        self.iter += 1

        if self.SumRate > self.max_sumrate:
            self.max_sumrate = self.SumRate

        # self.state = np.concatenate(
        #     (
        #         self.SINR,  # N
        #         [self.SumRate],  # 1
        #     ),
        #     axis=0,
        # )

        if self.state_dB:
            # tmp = []
            # for i in self.SINR:
            #     if i != 0:
            #         tmp.append(10 * log10(i))

            #     else:
            #         tmp.append(i)

            # self.state = np.array(tmp)

            self.state = np.array([log2(1 + i) for i in self.SINR])

        else:
            self.state = np.array(self.SINR)

        return self.state

    def Reset(self):
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])
        for i in enumerate(self.Users):
            i[1].w = Random_Complex_Mat(self.N, 1)
            i[1].w = i[1].w / np.linalg.norm(i[1].w)
            i[1].w = i[1].w * sqrt(self.transmitted_power * i[1].allocated_power)

        return self.State()

    def Step(self, action):
        self.CalculateSINR()
        old_sumrate = self.SumRate
        old_max = self.max_sumrate

        action = np.array(action)

        if not self.without_irs:
            self.Psi1 = np.diag(RealToPhase(action[0 : self.M1]))
            self.Psi2 = np.diag(RealToPhase(action[self.M1 : self.M1 + self.M2]))

        action = np.append(action, 1 - action[-1])

        for u in enumerate(self.Users):
            u[1].w = RealToPhase(
                action[
                    self.M1
                    + self.M2
                    + (u[0] * self.N) : self.M1
                    + self.M2
                    + (u[0] * self.N)
                    + self.N
                ]
            )
            u[1].w = (u[1].w).reshape(self.N, 1)

            u[1].allocated_power = action[-2 + u[0]]

            u[1].w = u[1].w / np.linalg.norm(u[1].w)
            u[1].w = u[1].w * sqrt(self.transmitted_power * u[1].allocated_power)

        new_state = self.State()
        reward = self.Reward()

        # reward += 0.8 * (self.SumRate - old_max) + 0.2 * (self.SumRate - old_sumrate)
        # reward += 1 * (self.SumRate - old_max)
        # reward += 1 * (self.SumRate - old_sumrate)
        # if self.SumRate < old_sumrate:
        #     reward -= 0.2

        # elif self.SumRate > old_sumrate:
        #     reward += 0.2

        return new_state, reward, self.SumRate, self.SINR

    def copy(self):
        new_env = Environment(
            num_of_antennas=self.N,
            num_of_irs1=self.M1,
            num_of_irs2=self.M2,
            path_loss_exponent=self.path_loss_exponent,
            irs1_to_antenna=self.irs1_to_antenna,
            irs2_to_antenna=self.irs2_to_antenna,
            irs1_to_irs2=self.irs1_to_irs2,
            transmitted_power=self.transmitted_power,
            reward_function=self.reward_function,
            state_dB=self.state_dB,
            without_irs=self.without_irs,
        )

        users = []
        for user in self.Users:
            users.append(
                new_env.CreateUser(
                    distance_to_antenna=user.distance_from_antenna,
                    distance_to_irs1=user.distance_from_irs1,
                    distance_to_irs2=user.distance_from_irs2,
                    noise_var=user.noise_power,
                    los_to_antenna=user.los_to_antenna,
                    los_to_irs1=user.los_to_irs1,
                    los_to_irs2=user.los_to_irs2,
                    sinr_threshold=user.sinr_threshold,
                    penalty=user.penalty,
                    allocated_power=user.allocated_power,
                    weight=user.weight,
                )
            )

        new_env.Psi1 = self.Psi1.copy()
        new_env.Psi2 = self.Psi2.copy()
        new_env.Hs1 = self.Hs1.copy()
        new_env.Hs2 = self.Hs2.copy()
        new_env.H12 = self.H12.copy()
        new_env.H21 = self.H21.copy()

        for i in enumerate(users):
            i[1].w = self.Users[i[0]].w.copy()
            i[1].hsu = self.Users[i[0]].hsu.copy()
            i[1].h1u = self.Users[i[0]].h1u.copy()
            i[1].h2u = self.Users[i[0]].h2u.copy()

        return new_env
