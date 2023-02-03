from Functions import *
from User import User
from cmath import sqrt
from math import log2, log10


class Environment:
    def __init__(self, *, num_of_antennas: int, num_of_irs1: int, num_of_irs2: int,
                 path_loss_exponent: float, irs1_to_antenna: float,
                 irs2_to_antenna: float, irs1_to_irs2: float) -> None:

        self.N = num_of_antennas  # Number of Antennas
        self.M1 = num_of_irs1  # Number of Elements of IRS1
        self.M2 = num_of_irs2  # Number of Elements of IRS2

        self.path_loss_exponent = path_loss_exponent
        self.irs1_to_antenna = irs1_to_antenna  # The Distance Between IRS1 & Antenna
        self.irs2_to_antenna = irs2_to_antenna  # The Distance Between IRS2 & Antenna
        self.irs1_to_irs2 = irs1_to_irs2  # The Distance Between IRS1 & IRS2

        self.Users = []
        self.SINR = []
        self.SumRate = 0
        self.num_of_users = 0

        # Generate Random Channel Coefficient Matrix(es)
        self.Hs1 = Random_Complex_Mat(self.M1, self.N) / self.irs1_to_antenna
        self.Hs2 = Random_Complex_Mat(self.M2, self.N) / self.irs2_to_antenna
        self.H12 = Random_Complex_Mat(self.M2, self.M1) / self.irs1_to_irs2
        self.H21 = np.conjugate(np.transpose(self.H12))

        # Generate Initial IRS Coefficient Matrix(es)
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])

    def CreateUser(self, *, distance_to_antenna: float, distance_to_irs1: float,
                   distance_to_irs2: float, noise_var: float, los_to_antenna: bool,
                   los_to_irs1: bool, los_to_irs2: bool, sinr_threshold: float,
                   penalty: float, allocated_power: float):

        Usr = User(distance_to_antenna, distance_to_irs1, distance_to_irs2,
                   noise_var, los_to_antenna, los_to_irs1, los_to_irs2,
                   sinr_threshold, penalty, allocated_power)

        Usr.GenerateMatrixes(self)
        self.Users.append(Usr)
        self.SINR.append(0)
        self.num_of_users += 1
        return Usr

    def CalculateSINR(self):
        SINR = []
        for i in enumerate(self.Users):
            numerator = np.absolute(
                (
                    i[1].hsu
                    + (i[1].h1u @ self.Psi1) @ self.Hs1
                    + (i[1].h2u @ self.Psi2) @ self.Hs2
                    + (i[1].h2u @ self.Psi2) @ self.Hs2
                    + ((i[1].h1u @ self.Psi1) @
                       self.H21) @ (self.Psi2 @ self.Hs2)
                    + ((i[1].h2u @ self.Psi2) @
                           self.H12) @ (self.Psi1 @ self.Hs1)
                ) @ i[1].w
            ) ** 2

            denominator = i[1].noise_power
            for j in enumerate(self.Users):
                if j[0] != i[0]:
                    denominator += np.absolute(
                        (
                            j[1].hsu
                            + (j[1].h1u @ self.Psi1) @ self.Hs1
                            + (j[1].h2u @ self.Psi2) @ self.Hs2
                            + (j[1].h2u @ self.Psi2) @ self.Hs2
                            + ((j[1].h1u @ self.Psi1) @
                               self.H21) @ (self.Psi2 @ self.Hs2)
                            + ((j[1].h2u @ self.Psi2) @
                               self.H12) @ (self.Psi1 @ self.Hs1)
                        ) @ j[1].w

                    ) ** 2

            SINR.append((numerator / denominator)[0, 0])

        self.SINR = [10*log10(i) for i in SINR]
        self.SumRate = sum(log2(1 + i) for i in SINR)

    def Reward(self) -> float:
        reward = self.SumRate
        for i in enumerate(self.SINR):
            if i[1] < self.Users[i[0]].sinr_threshold:
                reward -= self.Users[i[0]].penalty

        return reward

    def State(self) -> None:
        self.CalculateSINR()
        # self.state = np.concatenate(
        #     (
        #         self.SINR,  # N
        #         [self.SumRate],  # 1
        #     ),
        #     axis=0,
        # )
        self.state = np.array(self.SINR)
        return self.state

    def Reset(self):
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])
        for i in enumerate(self.Users):
            i[1].w = (Random_Complex_Mat(self.N, 1) /
                      sqrt(self.N)) * i[1].allocated_power

        return self.State()

    def Step(self, action):
        action = np.array(action)
        self.Psi1 = np.diag(RealToPhase(action[0: self.M1]))
        self.Psi2 = np.diag(RealToPhase(action[self.M1: self.M1 + self.M2]))

        for u in enumerate(self.Users):
            u[1].w = RealToPhase(
                action[self.M1 + self.M2 + (u[0] * self.N): self.M1 + self.M2+(u[0] * self.N) + self.N])
            u[1].w = (u[1].w).reshape(self.N, 1)
            u[1].w = (u[1].w / sqrt(self.N)) * u[1].allocated_power

        new_state = self.State()
        reward = self.Reward()

        return new_state, reward, self.SumRate, self.SINR
