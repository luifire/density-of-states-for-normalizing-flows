import numpy as np
import torch
from torch import linalg
from torch.distributions import MultivariateNormal
from warnings import warn

from master.config import *
from misc.constants import LOG_NO_INFLATION_VAR
from easy_access import EasyAccess

class NormalInflation:
    """Inflates a batch according to Horvat 2021 with Gaussian noise"""

    def __init__(self, normal_scale, tangent_scale, data_shape, rotator):
        self.data_shape = data_shape
        self.tangent_scale = tangent_scale
        self.normal_scale = normal_scale
        # not used for MNIST
        self.rotation_matrix = None if rotator is None else torch.from_numpy(rotator.rotation_matrix).cuda().float()
        # Horvat 2021 found that the radius of the sphere around the datapoints doesn't have a great impact
        # and they used this as default.
        #self.scale_factor = 0.5 * sigma**2

        # this is a special variance
        self.no_inflation = normal_scale is None or np.isclose(np.log10(normal_scale ** 2), LOG_NO_INFLATION_VAR)
        # self.go_down = scale is not None and np.isclose(np.log10(scale**2), 5)
        if not self.no_inflation:
            total_input_dimension = np.prod(data_shape)
            """if self.go_down:
                self.update_noise_generator(0)
            else:"""
            cov = normal_scale ** 2 * torch.eye(total_input_dimension, dtype=torch.float32, device=DEVICE)
            cov[0, 0] = tangent_scale ** 2
            self.noise_generator = MultivariateNormal(torch.zeros(total_input_dimension, dtype=torch.float32, device=DEVICE),
                                                      cov)

    def update_noise_generator(self, epoch):
        """
        if not self.go_down:
            return
        assert False

        total_input_dimension = np.prod(self.data_shape)
        total_epochs = EasyAccess().args.epochs

        if epoch > 0.7 * total_epochs:
            current_var = 1e-6
        else:
            start, end = -1, -6

            epoch_step = np.abs(end - start) / (0.7 * total_epochs)
            current_var = start - epoch * epoch_step
            print(f'Current log var {current_var:.3f}')
            current_var = 10 ** current_var

        self.noise_generator = MultivariateNormal(torch.zeros(total_input_dimension, dtype=torch.float32, device=DEVICE),
                                             current_var * torch.eye(total_input_dimension, dtype=torch.float32,
                                                                    device=DEVICE))
                                                                    """

    def __call__(self, batch):
        if self.no_inflation:
            return batch

        with torch.no_grad():
            noise = self.noise_generator.sample((len(batch),))

            # rotation
            if self.rotation_matrix is not None:
                rotated_noise = self.rotation_matrix @ noise.T
                noise = rotated_noise.T

            noise = noise.reshape((-1, *self.data_shape))

            return noise + batch


class UniformInflation:
    """Adds Uniform noise s.t. the signal/noise ratio is like
    that of of a given normal distribution"""
    def __init__(self, related_normal_noise_var):
        """3 * std dev contains 99% of the noise. Thus we should shrink
        the uniform noise to a range of 6 std dev (positive and negative area)
        checked: 1"""
        #self.range_normal_noise = 6 * np.sqrt(related_normal_noise_var)

        """sqrt(3) * std_dev results in the same variance distribution"""
        self.range_normal_noise = np.sqrt(3) * np.sqrt(related_normal_noise_var)

    def __call__(self, x):
        with torch.no_grad():
            noise = self.range_normal_noise * (torch.rand_like(x) - .5)
            return x + noise

    def update_noise_generator(self, epoch):
        pass

