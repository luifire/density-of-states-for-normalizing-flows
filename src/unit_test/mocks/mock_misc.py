import torch

from easy_access import EasyAccess
from mocks.utils import count_calls


@count_calls
def mock_loss_fn(_, x):
    def mean_mock():
        dummy = x.mean()
        # dummy backward function
        dummy.backward = lambda: None
        return dummy

    res = x.sum(1).sum(1).sum(1)
    res.mean = mean_mock

    return res
