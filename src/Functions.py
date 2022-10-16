import numpy as np
import cmath
import math
import random


def RealToPhase(input):
    tmp = [cmath.exp(complex(0, i)) for i in input]
    return np.array(tmp)


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
