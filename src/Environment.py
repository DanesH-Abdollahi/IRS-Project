from Functions import *
from User import User


class Environment:
    def __init__(self) -> None:
        self.N = 10  # Number of Antennas
        self.M1 = 10  # Number of Elements of IRS1
        self.M2 = 10  # Number of Elements of IRS2

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
        SINRThreshold: float,
        Penalty: float,
    ):
        Usr = User(d1, d2, d3, NoiseVar, LosToAntenna,
                   LosToIrs1, LosToIrs2, SINRThreshold, Penalty)
        Usr.GenerateMatrixes(self)
        self.Users.append(Usr)
        return Usr

    def Reward(self) -> float:
        self.CalculateSINR(self.state)
        reward = self.SumRate
        # Penalty = 2
        for i in enumerate(self.SINR):
            if i[1] < self.Users[i[0]].SINRThreshold:
                reward -= self.Users[i[0]].Penalty

        return reward

    def CalculateSINR(self, state):
        SINR = []
        self.Psi1 = RealToPhase(state[0:self.M1])
        self.Psi2 = RealToPhase(state[self.M1:self.M1 + self.M2])
        self.Psi1 = np.diag(self.Psi1)
        self.Psi2 = np.diag(self.Psi2)

        for u in enumerate(self.Users):
            u[1].w = RealToPhase(
                state[self.M1 + self.M2 + (u[0]*self.N):self.M1 + self.M2+(u[0]*self.N)+self.N])
            u[1].w = (np.reshape(u[1].w,  (self.N, 1)) /
                      cmath.sqrt(self.N)) * 100

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
                        i[1].w
                    )
                )
                ** 2
            )

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

        tmp = np.angle(self.Users[0].w[:, 0], deg=False)
        for i in range(1, len(self.Users)):
            tmp = np.concatenate(
                (tmp, np.angle(self.Users[i].w[:, 0], deg=False)))

        self.state = np.concatenate(
            (
                np.angle(np.diag(self.Psi1), deg=False),  # M1
                np.angle(np.diag(self.Psi2), deg=False),  # M2
                # np.angle(self.Users[0].w[:, 0], deg=False),  # N
                # np.angle(self.Users[1].w[:, 0], deg=False),  # N
                tmp,
                [0 for _ in range(len(self.Users) + 1)]
            ),
            axis=0,
        )

        self.CalculateSINR(self.state)
        self.state = np.concatenate(
            (
                np.angle(np.diag(self.Psi1), deg=False),  # M1
                np.angle(np.diag(self.Psi2), deg=False),  # M2
                # np.angle(self.Users[0].w[:, 0], deg=False),  # N
                # np.angle(self.Users[1].w[:, 0], deg=False),  # N
                tmp,
                self.SINR,  # Users Num.
                [self.SumRate]  # 1
            ),
            axis=0,
        )
        return self.state

    def reset(self):
        self.Psi1 = np.diag(Random_Complex_Mat(1, self.M1)[0])
        self.Psi2 = np.diag(Random_Complex_Mat(1, self.M2)[0])
        for i in enumerate(self.Users):
            i[1].w = (Random_Complex_Mat(self.N, 1) /
                      cmath.sqrt(self.N)) * 100

        return self.State()

    def step(self, action):

        reward = self.Reward()
        self.state = np.concatenate(
            (
                action[0:self.M1],  # M1
                action[self.M1:self.M1 + self.M2],  # M2
                # action[self.M1 + self.M2: self.M1 + self.M2 + self.N],  # N
                # action[self.M1 + self.M2 + self.N: self.M1 + \
                #        self.M2 + 2 * self.N],  # N
                action[self.M1 + self.M2: self.M1 + \
                       self.M2 + (len(self.Users) + 1) * self.N],
                self.SINR,
                [self.SumRate]
            ),
            axis=0,
        )

        return self.state, reward, self.SumRate, self.SINR
