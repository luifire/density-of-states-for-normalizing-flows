from mocks.utils import count_calls


class MockedOptimizer:

    @count_calls
    def zero_grad(self):
        pass

    @count_calls
    def step(self):
        pass

    @count_calls
    def state_dict(self):
        return dict()

    def get_call_count(self):
        return {'zero_grad': self.zero_grad.call_count,
                'step': self.step.call_count,
                'state_dict': self.state_dict.call_count}