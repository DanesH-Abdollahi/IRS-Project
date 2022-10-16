from Functions import *


class User:
    def __init__(self, d1: float, d2: float, d3: float, NoiseVar: float, LosToAntenna: bool,
                 LosToIrs1: bool, LosToIrs2: bool, SINR_Threshold: float, Penalty: float) -> None:

        self.DistanceFromAntenna = d1
        self.DistanceFromIrs1 = d2
        self.DistanceFromIrs2 = d3
        self.NoisePower = NoiseVar

        self.LosToAntenna = LosToAntenna
        self.LosToIrs1 = LosToIrs1
        self.LosToIrs2 = LosToIrs2
        self.Penalty = Penalty
    
        self.hsu = 0
        self.h1u = 0
        self.h2u = 0
        self.w = 0

        self.SINRThreshold = SINR_Threshold  # 6 dB approximately

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

        self.w = (Random_Complex_Mat(env.N, 1) / cmath.sqrt(env.N)) * 100
