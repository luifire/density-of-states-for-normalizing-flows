import numpy as np
import torch
from scipy.stats import uniform
from collections.abc import Iterable
import warnings

from master.data.data_helper import *
from master.data.rotated_shapes import RotatedData
import fqs

class SmoothstepData(RotatedData):
    """We use https://en.wikipedia.org/wiki/Smoothstep """
    smoothstep_rotation_seed = 271193  # SSIWEI92
    train_sample_seed = 3102019
    test_sample_seed = 1234567

    def __init__(self, start, end, offset, Range, growth,
                 rotation_seed=smoothstep_rotation_seed, **kwargs):

        self.start, self.end, self.offset, self.Range, self.growth = start, end, offset, Range, growth

        self.Z = self.integral_for_Z()

        super(SmoothstepData, self).__init__(rotation_seed=rotation_seed, **kwargs)

    def get_embedding_dimensionality(self):
        return 1

    def get_manifold_bounds(self):
        return self.start, self.end

    def integral_for_Z(self):
        """
        S = Start, E = End, A = Offset, R = Range, k = growth
        p(x) = 1/Z ( R * s(gx) + A), x in [S, E]
        Z = int ( R * s(gx) + A ) x in [S, E]

        s(x) = 0 if x < 0
        1 if x > 1
        f(x) if x in [0, 1]
        =>
        gx > 1 => x > 1/g
        =>
        int Zp(x) dx = int 0 for x < 0 +
        int f(x) for x in [0, 1/g] +
        R int 1 for x in [1/g, E] +
        int A for x in [S, E]"""

        start, end, offset, Range, growth = self.start, self.end, self.offset, self.Range, self.growth
        return offset * (end - start) + Range * (end - 1 / growth) + Range / growth * 1 / 2

    def cdf(self, X):
        start, end, offset, Range, growth = self.start, self.end, self.offset, self.Range, self.growth

        X, was_iterable = make_iterable(X)

        cdf = np.zeros_like(X)
        for i, x in enumerate(X):
            if x > end:
                raise Exception('Out of Range')

            cdf[i] = offset * (x - start)
            if x > 0:
                mx = middle_part = min(x, 1 / growth)
                # cdf[i] += Range * (growth**5 * mx**6 - 3 * growth**4 * mx**5 + 5/2 * growth**3 * mx**4)
                cdf[i] += Range * (- 1 / 2 * growth ** 2 * mx ** 3 * (growth * mx - 2))

                if x > 1 / growth:
                    cdf[i] += Range * (x - 1 / growth)  # unsure

        cdf = cdf / self.Z
        if was_iterable:
            return cdf
        else:
            return cdf[0]

    def _inverse_cdf_ex(self, u):
        start, end, offset, Range, growth = self.start, self.end, self.offset, self.Range, self.growth
        Z = self.Z
        if u <= self.cdf(0):  # 1/Z * offset * (-start):
            return u * Z / offset + start
        elif u >= self.cdf(1 / growth):
            return 1 / (offset + Range) * (u * Z - Range / (2 * growth) + offset * start + Range / growth)
        else:
            # this turns out to be a quartic problem
            # ax^4 + bx^3 + cx**2 + dx + e
            a = - Range * growth ** 3 / 2
            b = Range * growth ** 2
            c = 0
            d = offset
            e = -(Z * u + offset * start)

            roots = fqs.quartic_roots([a, b, c, d, e]).flatten()
            # find the only reasonable solution (not imaginary / between 0 and 1/g)
            solution = None
            for root in roots:
                if root.imag == 0.:
                    real = root.real
                    if 0. < real < 1 / growth:
                        assert solution == None
                        solution = real
            return solution

    def inverse_cdf(self, U):
        U, was_iterable = make_iterable(U)
        X = []
        for u in U:
            assert 0 <= u <= 1

            x = self._inverse_cdf_ex(u)
            X.append(x)

        if was_iterable:
            return np.array(X)
        else:
            return X[0]

    def sample_on_manifold(self, size):
        U = uniform.rvs(loc=0, scale=1, size=size)
        return self.inverse_cdf(U)

    def pdf_on_manifold(self, x):
        x, was_iterable = make_iterable(x)

        pds = np.zeros_like(x)
        relevant_x = np.logical_and(x >= self.start, x <= self.end)

        pds[relevant_x] = 1 / self.Z * (self.Range * SmoothstepData.smoothstep(self.growth * x[relevant_x]) + self.offset)
        # previously we did this
        #pds[x < self.start] = 0
        #pds[x > self.end] = 0

        if was_iterable is False:
            pds = pds[0]
        return pds

    @staticmethod
    def smoothstep(x):
        res = np.ones_like(x) * -1
        res[x <= 0] = 0
        res[x >= 1] = 1

        between_0_1 = np.logical_and(x > 0, x < 1)
        y = x[between_0_1]
        #res[between_0_1] = 3 * y ** 2 - 2 * y ** 3
        # to speed things up
        res[between_0_1] = y**2 * (3 - 2*y)

        return res

    def _embed_data(self, data, destination):
        destination[0] = data

    def _sample_for_plot(self, size):
        return np.linspace(self.start, self.end, size)

    def log_pdf_on_manifold(self, samples):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.log(self.pdf_on_manifold(samples))

    """def get_full_space_bounds(self):

        # same as for 1e-6
        used_scale = 1e-3 if self.normal_noise_scale is None else self.normal_noise_scale

        #log_scale = np.log10(used_scale ** 2)
        #parts_of_std_dev = 5 if self.dimensions == 2 else 4
        parts_of_std_dev = 5

        x_offset = (parts_of_std_dev+1) * used_scale + 0.25
        x_length = self.end - self.start + 2 * x_offset

        y_range = parts_of_std_dev * used_scale
        y_length = 2 * y_range

        x_granularity = (1 / x_length) / 4
        y_granularity = y_length / 2000

        self.y_granularity_of_grid_plot = y_granularity

        x_start, x_end = self.start - x_offset, self.end + x_offset

        x = RotatedData.arange(x_start, x_end, x_granularity)
        y = np.arange(-y_range, y_range, y_granularity)

        x_zero = y_zero = np.array([0])
        y_std_dev_1 = np.array([0])
        if not self.fine_grid:
            # remove all that are close to zero, as we cover this part separately
            # keep it, allows for better plotting
            #y = y[np.logical_not(np.isclose(y, 0))]

            x_zero = RotatedData.arange(x_start, x_end, x_granularity / 50)
            y_zero = np.array([0])

            y_std_dev_1 = np.array([1 * used_scale])

        # (x_bigger, y_bigger),
        return (x, y), (x_zero, y_zero), (x_zero, y_std_dev_1), (x_zero, -y_std_dev_1)"""

    def get_name(self):
        return 'Smooth Step'