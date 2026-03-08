import numpy as np
import torch
from scipy.stats import vonmises
import warnings

from master.data.rotated_shapes import RotatedData
from master.data.data_helper import *


class VonMises(RotatedData):
    sphere_rotation_seed = 51090
    train_sample_seed = 9031963
    test_sample_seed = 11031960

    # sphere initialisation
    def __init__(self, sphere_radius, mises_kappa, rotation_seed=sphere_rotation_seed, **kwargs):
        # if sphere_radius == None or 0: no radius embedding
        self.sphere_radius = sphere_radius
        self.mises_kappa = mises_kappa

        super(VonMises, self).__init__(rotation_seed=rotation_seed, **kwargs)

    def get_embedding_dimensionality(self):
        return 1

    def get_manifold_bounds(self):
        return -np.pi, np.pi

    def _embed_data(self, data, destination):
        if self.sphere_radius == 0:
            destination[0] = data
        else:
            s1 = self.sphere_radius * np.cos(data), self.sphere_radius * np.sin(data)
            destination[0:2] = s1

    def sample_on_manifold(self, size):
        return vonmises.rvs(self.mises_kappa, size=size)

    def _sample_for_plot(self, size):
        #return np.linspace(vonmises.ppf(0.01, kappa=self.mises_kappa), vonmises.ppf(0.99, kappa=self.mises_kappa), size)
        return np.linspace(-np.pi, np.pi, size)

    def log_pdf_on_manifold(self, samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.log(self.pdf_on_manifold(samples))

    def pdf_on_manifold(self, samples):
        pds = vonmises.pdf(samples, kappa=self.mises_kappa)
        pds, was_iterable = make_iterable(pds)
        pds[samples < -np.pi] = 0
        pds[samples > np.pi] = 0

        return pds if was_iterable else pds[0]

    def get_full_space_bounds(self):

        if self.normal_noise_scale is None:
            x_offset = 3
            y_range = 0.3
            granularity = 0.005
        else:
            log_scale = np.log10(self.normal_noise_scale ** 2)
            if log_scale == -2:
                x_offset = 0.5
                y_range = 1.5
                granularity = 0.02
            elif log_scale <= -4:
                x_offset = 0.5
                y_range = 1.5
                granularity = 0.02
            elif log_scale <= -0.5:
                x_offset = 4.5
                y_range = 4
                granularity = 0.04
            elif log_scale <= 0:
                x_offset = 5
                y_range = 6
                granularity = 0.06
            else:
                x_offset = 6
                y_range = 7
                granularity = 0.05

        """if self.noise_scale is None:
            x_offset = 3
            y_range = 0.3
            granularity = 0.005*2
        else:
            log_scale = np.log10(self.noise_scale**2)
            if log_scale <= -4:
                x_offset = 3
                y_range = 1.2
                granularity = 0.01
            elif log_scale <= -2:
                x_offset = 2.5
                y_range = 3
                granularity = 0.01
            elif log_scale <= -0.5:
                x_offset = 4.5
                y_range = 4
                granularity = 0.015
            else:
                x_offset = 7
                y_range = 7
                granularity = 0.03"""

        x = np.arange(-np.pi - x_offset, np.pi + x_offset, granularity)
        y = np.arange(-y_range, +y_range, granularity)

        return x, y

    def get_name(self):
        return 'von Mises'