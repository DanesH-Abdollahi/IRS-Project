import numpy as np
from math import e, pi


def RealToPhase(input):
    tmp = [1j * i for i in input]
    return e ** (np.array(tmp))


def Random_Complex_Mat(Row: int, Col: int):
    Matrix = np.random.uniform(low=-pi, high=pi, size=(Row, Col))
    return e ** (Matrix * 1j)
