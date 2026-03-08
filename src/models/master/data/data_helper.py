from collections import Iterable
import numpy as np
from contextlib import contextmanager


def make_iterable(X):
    if isinstance(X, Iterable):
        return X, True
    else:
        return np.array([X]), False


@contextmanager
def random_state_keeper(new_seed):
    try:
        random_state = np.random.get_state()
        np.random.seed(new_seed)
        yield random_state
    finally:
        np.random.set_state(random_state)
