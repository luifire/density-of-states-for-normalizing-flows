import torch

from mocks.utils import count_calls

class MockModel():

    @count_calls
    def train(self):
        pass

    @count_calls
    def eval(self):
        pass

    @count_calls
    def state_dict(self):
        return dict()

    @count_calls
    def sample(self, num_samples, manipulate_samples=True):
        return torch.zeros((num_samples, 1, 10, 10))
