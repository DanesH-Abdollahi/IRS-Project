from re import S
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

        # Generate Matrixes for User
        if Usr.LosToAntenna:
            Usr.hsu = Random_Complex_Mat(1, self.N) / Usr.DistanceFromAntenna

        if Usr.LosToIrs1:
            Usr.h1u = Random_Complex_Mat(1, self.M1) / Usr.DistanceFromIrs1

        if Usr.LosToIrs2:
            Usr.h2u = Random_Complex_Mat(1, self.M2) / Usr.DistanceFromIrs2

        Usr.w = Random_Complex_Mat(self.N, 1)
        self.Users.append(Usr)
        return Usr

    def SumRate(self):
        # Sumrate =
        pass

    def Reward(state):
        pass


class Agent:
    def __init__(self, env: Environment):
        self.Env = env

    def TakeAction(self):
        pass


def CalculateSINR(env: Environment):
    SINR = []
    for i in enumerate(env.Users):
        numerator = (
            np.absolute(
                np.dot(
                    i[1].hsu
                    + np.dot(i[1].h1u, np.dot(env.Psi1, env.Hs1))
                    + np.dot(i[1].h2u, np.dot(env.Psi2, env.Hs2))
                    + np.dot(i[1].h2u, np.dot(env.Psi2, env.Hs2))
                    + np.dot(
                        np.dot(i[1].h1u, np.dot(env.Psi1, env.H21)),
                        np.dot(env.Psi2, env.Hs2),
                    )
                    + np.dot(
                        np.dot(i[1].h2u, np.dot(env.Psi2, env.H12)),
                        np.dot(env.Psi1, env.Hs1),
                    ),
                    i[1].w,
                )
            )
            ** 2
        )

        denominator = i[1].NoiseVar
        for j in enumerate(env.Users):
            if j[0] != i[0]:
                denominator += (
                    np.absolute(
                        np.dot(
                            j[1].hsu
                            + np.dot(j[1].h1u, np.dot(env.Psi1, env.Hs1))
                            + np.dot(j[1].h2u, np.dot(env.Psi2, env.Hs2))
                            + np.dot(j[1].h2u, np.dot(env.Psi2, env.Hs2))
                            + np.dot(
                                np.dot(j[1].h1u, np.dot(env.Psi1, env.H21)),
                                np.dot(env.Psi2, env.Hs2),
                            )
                            + np.dot(
                                np.dot(j[1].h2u, np.dot(env.Psi2, env.H12)),
                                np.dot(env.Psi1, env.Hs1),
                            ),
                            j[1].w,
                        )
                    )
                    ** 2
                )
        SINR.append((i[1].h))

    return SINR


def Run():
    env = Environment()
    U1 = env.CreateUser(17.5, 10, 10, 1, False, True, False)
    agent = Agent(env)

    # env = Environment(u1)
    print(abs(U1.w))
    # print(abs(env.h1u1))
    # u2 = Environment.User(23, 15, 15)
    # a = Environment(u1, u2)


if __name__ == "__main__":
    Run()
