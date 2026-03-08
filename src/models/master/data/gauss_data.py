import numpy as np
import torch
from scipy.stats import norm
import warnings

from master.data.rotated_shapes import RotatedData
from master.data.data_helper import *


class GaussData(RotatedData):
    sphere_rotation_seed = 51090

    # sphere initialisation
    def __init__(self, **kwargs):
        super(GaussData, self).__init__(rotation_seed=0, **kwargs)

    def get_embedding_dimensionality(self):
        return 1

    def get_manifold_bounds(self):
        warnings.warn('incorrect manifold bounds')
        return -2 * self.normal_noise_scale, 2 * self.normal_noise_scale

    def _embed_data(self, data, destination):
        destination[0] = data

    def sample_on_manifold(self, size):
        return np.zeros(size)

    def _sample_for_plot(self, size):
        #return np.linspace(vonmises.ppf(0.01, kappa=self.mises_kappa), vonmises.ppf(0.99, kappa=self.mises_kappa), size)
        return np.linspace(-2 * self.normal_noise_scale, 2 * self.normal_noise_scale, size)

    def log_pdf_on_manifold(self, samples):
        return np.zeros(len(samples))

    def log_pdf_on_inflated_manifold(self, inflated_manifold):
        return norm.logpdf(inflated_manifold, scale=self.normal_noise_scale)

    def pdf_on_manifold(self, samples):
        return np.ones(len(samples))

    def get_full_space_bounds(self):

        if self.normal_noise_scale is None:
            x_offset = 3
            y_range = 0.3
            granularity = 0.005*5
        else:
            log_scale = np.log10(self.normal_noise_scale ** 2)
            if log_scale <= -4:
                x_offset = 3
                y_range = 1.2
                granularity = 0.02
            elif log_scale <= -2:
                x_offset = 2.5
                y_range = 3
                granularity = 0.03
            elif log_scale <= -0.5:
                x_offset = 4.5
                y_range = 4
                granularity = 0.035
            else:
                x_offset = 7
                y_range = 7
                granularity = 0.07

        x = np.arange(-np.pi - x_offset, np.pi + x_offset, granularity)
        y = np.arange(-y_range, +y_range, granularity)

        return x, y
