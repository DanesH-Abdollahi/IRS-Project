import numpy as np
import math
import random
from math import e, pi


def RealToPhase(input):
    tmp = [1j * i for i in input]
    return e ** (np.array(tmp))


def Random_Complex_Mat(Row: int, Col: int):
    Matrix = np.random.uniform(low=-pi, high=pi, size=(Row, Col))
    return e ** (Matrix * 1j)

# def Random_Complex_Mat(Row: int, Col: int):
#     tmp = []
#     for _ in range(Row):
#         tmp.append(
#             [e ** (1j * random.uniform(-math.pi, math.pi)) for _ in range(Col)]
#         )
#     Matrix = np.array(tmp)
#     Matrix = Matrix.reshape(Row, Col)
#     return Matrix
