from Functions import *
from cmath import sqrt


class User:
    def __init__(self, d_from_antenna: float, d_from_irs1: float, d_from_irs2: float,
                 noise_var: float, los_to_antenna: bool, los_to_irs1: bool, los_to_irs2: bool,
                 sinr_threshold: float, penalty: float, allocated_power: float, weight: float) -> None:

        self.distance_from_antenna = d_from_antenna
        self.distance_from_irs1 = d_from_irs1
        self.distance_from_irs2 = d_from_irs2
        self.noise_power = noise_var

        self.los_to_antenna = los_to_antenna   # Line of sight to antenna
        self.los_to_irs1 = los_to_irs1         # Line of sight to IRS1
        self.los_to_irs2 = los_to_irs2         # Line of sight to IRS2
        self.penalty = penalty                 # Penalty for not meeting SINR threshold
        self.allocated_power = allocated_power # Power allocated to user
        self.sinr_threshold = sinr_threshold   # dB
        self.weight = weight                   # Weight of user

        self.hsu = 0
        self.h1u = 0
        self.h2u = 0
        self.w = 0

    def GenerateMatrixes(self, env) -> None:
        if self.los_to_antenna:
            self.hsu = Random_Complex_Mat(
                1, env.N) / self.distance_from_antenna
        else:
            self.hsu = np.zeros((1, env.N))

        if self.los_to_irs1:
            self.h1u = Random_Complex_Mat(1, env.M1) / self.distance_from_irs1

        else:
            self.h1u = np.zeros((1, env.M1))

        if self.los_to_irs2:
            self.h2u = Random_Complex_Mat(1, env.M2) / self.distance_from_irs2

        else:
            self.h2u = np.zeros((1, env.M2))

        # self.w = (Random_Complex_Mat(env.N, 1) /
        #           sqrt(env.N)) * self.allocated_power

        self.w = Random_Complex_Mat(env.N, 1)
