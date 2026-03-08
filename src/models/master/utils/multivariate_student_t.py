"""Copied and adjusted from:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py
"""

import torch
import numpy as np

# Master addition
from master.config import DEVICE
from master.utils.computational import SampleManipulator
from easy_access import EasyAccess

from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import _batch_mahalanobis
from scipy.special import loggamma

class SimpleStudentT(Distribution):
    """Assuming isotropic cov matrix"""

    def __init__(self, degrees_of_freedom, dimensionality):
        v = self.v = self.degrees_of_freedom = degrees_of_freedom
        d = self.d = self.dimensionality = dimensionality

        self.log_constant_factor = torch.tensor(loggamma((v+d) / 2) - loggamma(v/2)
                                                - d/2 * np.log(v) - d/2 * np.log(np.pi), device=DEVICE)
        self.cov = torch.eye(d).to(DEVICE)

    def log_prob(self, value, mu=None):
        if mu is not None:
            centered_val = value - mu
        else:
            centered_val = value

        # copied from torch.multivariate_normal
        M = _batch_mahalanobis(self.cov, centered_val)

        log_energy = -(self.v + self.d)/2 * torch.log(1 + M)
        log_prob = self.log_constant_factor + log_energy
        return log_prob
