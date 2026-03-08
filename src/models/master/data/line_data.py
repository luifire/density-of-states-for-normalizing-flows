from scipy.stats import uniform
import numpy as np
import torch
import warnings

from master.data.rotated_shapes import RotatedData
from master.data.data_helper import *
from misc.common_functions import pdf_N_plus_Uni
from master.utils.utils import beep

class LineData(RotatedData):
    line_rotation_seed = 42
    train_sample_seed = 27081993
    test_sample_seed = 26021995

    # line initialisation
    def __init__(self, line_length, rotation_seed=line_rotation_seed,
                 normal_noise_scale=None, **kwargs):

        self.line_length = line_length
        self.start = -line_length / 2
        self.end = self.start + self.line_length
        self.pdf = 1 / line_length  # 1/line_length
        self.var = normal_noise_scale ** 2 if normal_noise_scale is not None else None

        super(LineData, self).__init__(rotation_seed=rotation_seed,
                                       normal_noise_scale=normal_noise_scale, **kwargs)

    def get_embedding_dimensionality(self):
        return 1

    def get_manifold_bounds(self):
        return self.start, self.start + self.line_length

    def log_pdf_on_manifold(self, samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.log( self.pdf_on_manifold(samples) )

    def log_pdf_on_inflated_manifold(self, samples):
        if self.var is None:
            return self.log_pdf_on_manifold(samples)
        else:
            return np.log(pdf_N_plus_Uni(samples, self.var))

    def pdf_on_manifold(self, samples):
        pds = np.ones_like(samples) * self.pdf

        pds, was_iterable = make_iterable(pds)

        # 0 outside given region
        pds[np.logical_or(samples < self.start, samples > self.end)] = 0
        return pds if was_iterable else pds[0]

    def _embed_data(self, data, destination):
        destination[0] = data

    def sample_on_manifold(self, size):
        return uniform.rvs(loc=self.start, scale=self.line_length, size=size)

    def _sample_for_plot(self, size):
        return np.linspace(self.start, self.end, size)

    def get_full_space_bounds(self):
        used_scale = 1e-3 if self.normal_noise_scale is None else self.normal_noise_scale

        # parts_of_std_dev = 5 if self.dimensions == 2 else 4
        parts_of_std_dev = 5

        x_offset = (parts_of_std_dev + 1) * used_scale + 0.25
        x_length = self.end - self.start + 2 * x_offset

        y_range = parts_of_std_dev * used_scale
        y_length = 2 * y_range

        x_granularity = (1 / x_length) / 2
        y_granularity = y_length / 2000  # #(1 / y_length) / 100000

        self.y_granularity_of_grid_plot = y_granularity

        x_start, x_end = self.start - x_offset, self.end + x_offset

        x = RotatedData.arange(x_start, x_end, x_granularity)
        y = np.arange(-y_range, y_range, y_granularity)

        x_zero = y_zero = np.array([0])
        y_std_dev_1 = np.array([0])
        if not self.fine_grid:
            # remove all that are close to zero, as we cover this part separately
            # keep it, allows for better plotting
            # y = y[np.logical_not(np.isclose(y, 0))]

            x_zero = RotatedData.arange(x_start, x_end, x_granularity / 50)
            y_zero = np.array([0])

            y_std_dev_1 = np.array([1 * used_scale])

        # (x_bigger, y_bigger),
        return (x, y), (x_zero, y_zero), (x_zero, y_std_dev_1), (x_zero, -y_std_dev_1)

        """if self.normal_noise_scale is None:
            x_offset = 0.2
            x_granularity = 0.005
            y_range = 0.1
            y_granularity = 0.001
            log_scale = None
        else:
            log_scale = np.log10(self.normal_noise_scale ** 2)
            x_offset = 5 * self.normal_noise_scale
            y_range = 5 * self.normal_noise_scale
            if log_scale <= -6:
                x_granularity = 0.005
                y_granularity = 0.0005
            elif log_scale <= -4:
                x_granularity = 0.005
                y_granularity = 0.001
            elif log_scale <= -2:
                x_granularity = 0.005
                y_granularity = 0.007
            elif log_scale <= -0.5:
                x_granularity = 0.008
                y_granularity = 0.02
            elif log_scale <= 0:
                x_granularity = 0.01
                y_granularity = 0.0200
            else:
                x_granularity = 0.03
                y_granularity = 0.02

        self.y_granularity_of_grid_plot = y_granularity

        x_start, x_end = self.start - x_offset, self.end + x_offset

        x = RotatedData.arange(x_start, x_end, x_granularity)
        y = np.arange(-y_range, y_range, y_granularity)
        # remove all that are close to zero, as we cover this part separately
        # keep it, allows for better plotting
        #y = y[np.logical_not(np.isclose(y, 0))]

        x_zero = RotatedData.arange(x_start, x_end, x_granularity / 10)
        y_zero = np.array([0])

        x_bigger = y_bigger = np.array([0])
        if log_scale is None or log_scale < -1:
            if self.fine_grid or self.dimensions > 2:
                y_granularity *= 20
                x_granularity *= 4
            else:
                y_granularity *= 10

            x_bigger = RotatedData.arange(x_start, x_end, x_granularity)
            y_bigger = np.concatenate((np.arange(-y_range-1 + y_granularity, -y_range, y_granularity),
                                       np.arange(y_range + y_granularity, y_range+1, y_granularity)))

        return (x, y), (x_bigger, y_bigger), (x_zero, y_zero)"""

    def get_name(self):
        return 'Uniform'