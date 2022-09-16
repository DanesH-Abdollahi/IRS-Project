import numpy as np
import cmath
import math
import random


def Random_Complex_Mat(Row: int, Col: int):
    tmp = []
    for _ in range(Row):
        tmp.append(
            [cmath.exp(complex(0, random.uniform(0, 2 * math.pi))) for _ in range(Col)]
        )
    Matrix = np.array(tmp)
    Matrix = Matrix.reshape(Row, Col)
    return Matrix


class User:
    def __init__(self, d1: float, d2: float, d3: float, NoiseVar: float) -> None:
        self.DistanceFromAntenna = d1
        self.DistanceFromIrs1 = d2
        self.DistanceFromIrs2 = d3
        self.NoisePower = NoiseVar

        self.LosToAntenna = True
        self.LosToIrs1 = True
        self.LosToIrs2 = True

    def LosToAntennaFunc(self, Trigger: bool) -> None:
        self.LosToAntenna = Trigger

    def LosToIrs1Func(self, Trigger: bool) -> None:
        self.LosToIrs1 = Trigger

    def LosToIrs2Func(self, Trigger: bool) -> None:
        self.LosToIrs2 = Trigger


class Environment:
    def __init__(self, User1: User) -> None:
        self.N = 5  # Number of Antennas
        self.M1 = 4  # Number of Elements of IRS1
        self.M2 = 4  # Number of Elements of IRS1

        self.PathLosExponent = 2
        self.Irs1ToAntenna = 10  # The Distance Between IRS1 & Antenna
        self.Irs2ToAntenna = 10  # The Distance Between IRS2 & Antenna
        self.Irs1ToIrs2 = 10  # The Distance Between IRS1 & IRS2

        # Generate Random Channel Coefficient Matrix(es)
        self.Hs1 = Random_Complex_Mat(self.M1, self.N) / self.Irs1ToAntenna
        self.Hs2 = Random_Complex_Mat(self.M2, self.N) / self.Irs2ToAntenna
        self.Hs12 = Random_Complex_Mat(self.M2, self.M1) / self.Irs1ToIrs2

        # Generate Matrixes for User 1
        if User1.LosToAntenna:
            self.hsu1 = Random_Complex_Mat(1, self.N) / User1.DistanceFromAntenna
        else:
            self.hsu1 = 0

        if User1.LosToIrs1:
            self.h1u1 = Random_Complex_Mat(1, self.M1) / User1.DistanceFromIrs1
        else:
            self.h1u1 = 0

        if User1.LosToIrs2:
            self.h2u1 = Random_Complex_Mat(1, self.M2) / User1.DistanceFromIrs2
        else:
            self.h2u1 = 0

        # Generate Matrixes for User 2
        # if User2.LosToAntenna:
        #     self.hsu2 = Random_Complex_Mat(1, self.N) / User2.DistanceFromAntenna
        # else:
        #     self.hsu2 = 0

        # if User2.LosToIrs1:
        #     self.h1u2 = Random_Complex_Mat(1, self.M1) / User2.DistanceFromIrs1
        # else:
        #     self.h1u2 = 0

        # if User2.LosToIrs2:
        #     self.h2u2 = Random_Complex_Mat(1, self.M2) / User2.DistanceFromIrs2
        # else:
        #     self.h2u2 =

        # Generate Initial IRS Coefficient Matrix(es)
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])
        # self.Psi1 = Random_Complex_Mat(1, self.M1)

    def Reward(state):
        pass


class Agent:
    def __init__(self):
        pass

    def TakeAction(self):
        pass


def Run():
    u1 = User(17.5, 10, 10, 1)
    u1.LosToAntennaFunc(False)
    u1.LosToIrs2Func(False)

    env = Environment(u1)
    print(abs(env.Psi1))
    # print(abs(env.h1u1))
    # u2 = Environment.User(23, 15, 15)
    # a = Environment(u1, u2)


if __name__ == "__main__":
    Run()
