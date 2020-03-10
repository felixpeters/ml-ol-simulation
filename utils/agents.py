import random

import numpy as np

def random_beliefs(m):
    a = np.random.randint(-1, 2, m)
    return a

def random_reality(m):
    a = np.random.randint(0, 2, m)
    a[a == 0] = -1
    return a

def random_dims(m, k):
    dims = [i for i in range(m)]
    return random.sample(dims, k)
