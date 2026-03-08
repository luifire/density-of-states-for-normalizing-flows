from scipy.stats import norm, uniform, multivariate_normal
import numpy as np


class NoiseWrapper:

    def __init__(self, dist, normal_noise_scale, tangent_noise_scale):
        self.dist = dist
        self.normal_noise_scale = normal_noise_scale

        if self.dist == 'uniform' and tangent_noise_scale is None:
            tangent_noise_scale = normal_noise_scale
        elif self.dist == 'uniform' and normal_noise_scale != tangent_noise_scale:
            raise NotImplementedError()

        self.tangent_noise_scale = tangent_noise_scale


    def _get_uniform_scale(self):
        if self.normal_noise_scale is not None:
            """3 * std dev contains 99% of the noise. Thus we should shrink
            the uniform noise to a range of 6 std dev (positive and negative area)
            checked: 1"""
            #return 6 * self.normal_noise_scale

            """sqrt(3) * std_dev results in the same variance"""
            return np.sqrt(3) * self.normal_noise_scale
        else:
            None

    def _get_uniform_loc(self):
        return -1/2 * self._get_uniform_scale()

    def _get_normal_cov_matrix(self, size):
        cov = self.normal_noise_scale ** 2 * np.eye(size)
        cov[0, 0] = self.tangent_noise_scale ** 2
        return cov

    def rvs_full_space(self, shape):
        if self.normal_noise_scale is None:
            return np.zeros(shape)
        else:
            if self.dist == 'uniform':
                return uniform.rvs(size=shape, loc=self._get_uniform_loc(), scale=self._get_uniform_scale())
            else:
                return multivariate_normal.rvs(cov=self._get_normal_cov_matrix(size=shape[1]), size=shape[0])
                #return norm.rvs(scale=self.normal_noise_scale, size=size)

    def rvs_tangent(self, size):
        if self.normal_noise_scale is None:
            return np.zeros(size)
        else:
            if self.dist == 'uniform':
                return uniform.rvs(size=size, loc=self._get_uniform_loc(), scale=self._get_uniform_scale())
            else:
                return norm.rvs(scale=self.tangent_noise_scale, size=size)

    def ambient_logpdf(self, data):
        if self.dist == 'uniform':
            return uniform.logpdf(x=data, loc=self._get_uniform_loc(), scale=self._get_uniform_scale()).sum(1)
        else:
            return multivariate_normal.logpdf(x=data, cov=self.normal_noise_scale ** 2 * np.eye(data.shape[1]))
            #return norm.logpdf(data, scale=self.normal_noise_scale)
