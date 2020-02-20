import numpy as np

def random_beliefs(k):
    a = np.random.randint(-1, 2, k)
    return a

def random_reality(k):
    a = np.random.randint(0, 2, k)
    a[a == 0] = -1
    return a
