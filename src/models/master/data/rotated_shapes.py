import numpy as np
import torch
from scipy.stats import norm, special_ortho_group, uniform
from scipy.special import logsumexp
from warnings import warn

from master.data.data_helper import *
from misc.constants import *
from probability_functions import *
from master.data.noise_wrapper import NoiseWrapper

from master.utils.utils import beep

"""
data_mode = [sampled,  full_grid, manifold_grid]
Function which gets manifold samples,
    adds noise to them
    rotates them
    compute llh
        manifold ?
        manifold + Noise
        fully inflated
    data
        manifold
        manifold with noise
        fully inflated

    data sampling:
        sampled
            along manifold
            inflation
        grid along manifold
            needed to plot pdf along manifold
        entire grid
            problem: beim Plotten hat man dann ein gedrehtes grid
                aber trotzdem schnell, wenn man batch processing macht und den Rest schneide ich ab.
"""

class RotatedData:
    FIXED_SAMPLE_SEED = 1337
    CONVOLVE_SAMPLE_COUNT = 1000
    #MANIFOLD_GRANULARITY = 0.005
    MANIFOLD_GRID_POINTS = 300

    def __init__(self, dimension, rotation_seed, data_mode, sample_seed=FIXED_SAMPLE_SEED,
                 normal_noise_scale=None, tangent_noise_scale=None, **kwargs):
        """data_mode: sampled, sampled_manifold, full_grid, manifold_grid"""

        self.dimensions = dimension
        self.normal_noise_scale = normal_noise_scale
        self.tangent_noise_scale = tangent_noise_scale
        self.sample_seed = sample_seed
        self.samples_to_delete = None
        noise_dist = 'uniform' if 'uniform_noise' in kwargs and kwargs['uniform_noise'] else 'norm'
        self.rotation_matrix = self._generate_rotation_matrix(rotation_seed)

        self.noise = NoiseWrapper(noise_dist, normal_noise_scale, tangent_noise_scale)
        RotatedData._kwargs_controller(kwargs)

        if data_mode == 'full_grid_fine':
            self.data_mode = 'full_grid'
            self.fine_grid = True
        else:
            self.data_mode = data_mode
            self.fine_grid = False

        # special variable for the grid plot case giving the granularity in y direction
        self.y_granularity_of_grid_plot = None

        self.random_manifold_samples = None
        generated_data = self.__generate_data(**kwargs)
        densities = self.__compute_densities(generated_data)

        generated_data, densities = self._remove_rare_event_samples(generated_data, densities)
        self.densities = densities

        generated_data[EMBEDDED_DATA] = self.__embed_data(generated_data[NONE_EMBEDDED_DATA])
        self.data = generated_data
        # need manifold samples and train samples

    def _remove_rare_event_samples(self, generated_data, densities):
        """There are rare events for which it is hard to determin the density by convolution
        we will remove these samples in this part"""
        if self.data_mode == 'full_grid' or self.samples_to_delete is None:
            # normal in this regime
            return generated_data, densities

        keeper = np.logical_not(self.samples_to_delete)
        if self.samples_to_delete.any():
            ...
        for key in generated_data.keys():
            if generated_data[key] is not None:
                generated_data[key] = generated_data[key][keeper]

        for key in densities.keys():
            densities[key] = densities[key][keeper]

        return generated_data, densities

    def _generate_rotation_matrix(self, rotation_seed):
        if rotation_seed is None:
            return np.identity(self.dimensions)

        with random_state_keeper(rotation_seed):
            # double checked - this always produces the same rotation matrix
            return special_ortho_group.rvs(self.dimensions)

    def __embed_data(self, data_to_embed):
        embedded_data = self.rotation_matrix @ data_to_embed.T
        embedded_data = embedded_data.T

        # make an image out of it
        image_dim = int(np.sqrt(self.dimensions))
        if image_dim ** 2 == self.dimensions:
            embedded_data = embedded_data.reshape((len(embedded_data), image_dim, image_dim))
        else:
            embedded_data = embedded_data.reshape((len(embedded_data), 1, self.dimensions))

        embedded_data = torch.from_numpy(embedded_data).unsqueeze(1).float()

        return embedded_data

    def __densities_for_full_grid_without_scaling(self, generated_data, infl_manifold_log_pd):
        """Special density computation for full grid when noise is 0."""
        full_data = generated_data[NONE_EMBEDDED_DATA]
        full_grid_densities = np.ones(len(full_data)) * -np.inf
        # initially all are set to true / this is a small hack to get that
        remaining_idx = full_grid_densities == -np.inf

        # go over all dimensions but skip the first dimension / this contains the manifold
        for dim in range(self.dimensions - 1):
            remaining_idx = np.logical_and(remaining_idx, np.isclose(full_data[:, dim + 1], 0))
        full_grid_densities[remaining_idx] = infl_manifold_log_pd[remaining_idx]
        return full_grid_densities

    def __compute_densities(self, generated_data):
        densities = dict()
        ########### manifold part
        inflated_manifold = generated_data[INFLATED_MANIFOLD_DATA]
        if self.normal_noise_scale is None:
            infl_manifold_log_pd = densities[INFLATED_MANIFOLD_LOG_PD] = self.log_pdf_on_manifold(inflated_manifold)
            if self.data_mode == 'full_grid':
                densities[FULL_INFLATED_LOG_PD] = \
                    self.__densities_for_full_grid_without_scaling(generated_data, infl_manifold_log_pd)
            else:
                densities[FULL_INFLATED_LOG_PD] = infl_manifold_log_pd

            return densities

        log_pd_infl_manifold = self.log_pdf_on_inflated_manifold(inflated_manifold)
        #if 'log_pdf_on_inflated_manifold' in dir(self):
        #    log_pd_infl_manifold = self.log_pdf_on_inflated_manifold(inflated_manifold)
        #else:
        #    log_pd_infl_manifold = self.compute_log_pd_on_inflated_manifold(inflated_manifold)

        densities[INFLATED_MANIFOLD_LOG_PD] = log_pd_infl_manifold

        ########### manifold + noise
        full_data = generated_data[NONE_EMBEDDED_DATA]
        data_without_manifold = full_data[:, 1:]
        #inflation_log_pd = norm.logpdf(data_without_manifold, scale=self.noise_scale).sum(1)
        inflation_log_pd = self.noise.ambient_logpdf(data_without_manifold)#.sum(1)

        densities[FULL_INFLATED_LOG_PD] = inflation_log_pd + log_pd_infl_manifold
        return densities

    def log_pdf_on_inflated_manifold(self, inflated_manifold):
        return self._compute_log_pd_on_inflated_manifold(inflated_manifold)

    def _compute_log_pd_on_inflated_manifold(self, samples):
        with random_state_keeper(91119):
            random_normal_samples = self.__get_normal_samples_for_convolution()
            #self.random_manifold_samples = self.sample_on_manifold(RotatedData.CONVOLVE_SAMPLE_COUNT)

        # due to memory constrains we need to split the samples here
        # np.tile and pdf_on_manifold seem to need a lot of memory
        BATCH_SIZE = 20000
        batch_count = len(samples) // BATCH_SIZE
        batch_count += 1 if batch_count == 0 else 0

        resulting_pds = []
        samples_to_delete = []  # there are really rare events that can occur for which it is hard to determin
        # the density with convolution. We will delete these samples
        for batch in np.array_split(samples, batch_count):
            # https://en.wikipedia.org/wiki/Convolution_of_probability_distributions
            repeated_samples = np.tile(batch, (RotatedData.CONVOLVE_SAMPLE_COUNT, 1)).T
            convolved_samples = repeated_samples - random_normal_samples
            pds = self.pdf_on_manifold(convolved_samples)
            mean = pds.mean(1)
            #if self.data_mode != 'full_grid' or any(mean < 0):
            """if any(mean == 0) or not np.isfinite(mean).all():
                raise OverflowError(f'Rare event. Consider deleting this sample. {self.dimensions}' +
                                    f'var {self.noise_scale**2} Data: {self}')"""
            samples_to_delete.append(np.logical_or(mean == 0, np.logical_not(np.isfinite(mean))))
            # this is fine with full grid
            mean[mean != 0] = np.log(mean[mean != 0])
            mean[mean == 0] = -np.inf

            resulting_pds.append(mean)
        self.samples_to_delete = np.concatenate(samples_to_delete)

        return np.concatenate(resulting_pds)

    def _get_inflated_manifold_bounds(self, sigma_count=2):
        left, right = self.get_manifold_bounds()
        if self.normal_noise_scale is not None:
            # covers 95% of the added noise
            left -= sigma_count * self.normal_noise_scale
            right += sigma_count * self.normal_noise_scale
        else:
            add_x_percent = .2 * (right - left)
            left -= add_x_percent
            right += add_x_percent

        return left, right

    def __generate_data(self, **kwargs):
        if self.data_mode == 'sampled':
            sample_size = kwargs['size']
            with random_state_keeper(self.sample_seed):
                manifold_data = self.sample_on_manifold(sample_size)
                inflation_data = self.noise.rvs_full_space(shape=(sample_size, self.dimensions))

            inflation_data[:, 0] += manifold_data

            whole_space = inflation_data
            inflated_manifold = inflation_data[:, 0]

        elif self.data_mode == 'sampled_manifold':
            sample_size = kwargs['size']
            with random_state_keeper(self.sample_seed):
                manifold_data = self.sample_on_manifold(sample_size)

            inflated_manifold = manifold_data + self.noise.rvs_tangent(size=sample_size)

            whole_space = np.zeros((sample_size, self.dimensions))
            whole_space[:, 0] = inflated_manifold
        elif self.data_mode == 'full_grid':
            # we already created data along the manifold.
            # Now we also create more points
            plotting_areas = self.get_full_space_bounds()

            slices = []
            for x, y in plotting_areas:
                xx, yy = np.meshgrid(x, y)
                point_pile = np.vstack([xx.ravel(), yy.ravel()]).T
                #point_pile = point_pile[::5]

                slices.append(point_pile)

            self.important_grid_part = (0, len(slices[0])-1)
            data_slice = np.concatenate(slices)

            print('Points in Grid:', len(data_slice))
            whole_space = np.zeros((len(data_slice), self.dimensions))
            whole_space[:, 0] = data_slice[:, 0]
            whole_space[:, 1] = data_slice[:, 1]

            manifold_data = None
            # only use the once along the manifold
            #inflated_manifold = whole_space[:, 0][np.isclose(whole_space[:, 1], 0)]
            inflated_manifold = whole_space[:, 0]

        elif self.data_mode == 'manifold_grid':
            left, right = self._get_inflated_manifold_bounds()
            #inflated_manifold = np.arange(left, right, RotatedData.MANIFOLD_GRANULARITY)
            inflated_manifold = np.linspace(left, right, RotatedData.MANIFOLD_GRID_POINTS)

            whole_space = np.zeros((len(inflated_manifold), self.dimensions))
            whole_space[:, 0] = inflated_manifold
            manifold_data = None

        return {'manifold': manifold_data,
                INFLATED_MANIFOLD_DATA: inflated_manifold,
                NONE_EMBEDDED_DATA: whole_space}

    def __get_normal_samples_for_convolution(self):
        return self.noise.rvs_tangent(size=RotatedData.CONVOLVE_SAMPLE_COUNT)
            # \
        #.reshape((RotatedData.CONVOLVE_SAMPLE_COUNT, -1))
        """return np.random\
            .normal(scale=self.noise_scale, size=RotatedData.CONVOLVE_SAMPLE_COUNT * self.get_embedding_dimensionality()) \
            .reshape((RotatedData.CONVOLVE_SAMPLE_COUNT, -1))"""

    def sample_on_manifold(self, size):
        raise Exception('Not implemented')

    def log_pdf_on_manifold(self, samples):
        raise Exception('Not implemented')

    def pdf_on_manifold(self, samples):
        raise Exception('Not implemented')

    def get_embedding_dimensionality(self):
        raise Exception('Not implemented')

    def get_manifold_bounds(self):
        raise Exception('Not implemented')

    def get_name(self):
        raise Exception('Not implemented')

    def get_printable_var(self):
        if self.normal_noise_scale is None:
            return f'$\mathcal{{N}}$(0, 0)' # σ²=
        else:
            return f'$\mathcal{{N}}$(0, ${{10^ {{{int(np.log10(self.normal_noise_scale**2))}}} }}$)'

    def get_name_with_var(self):
        return f'{self.get_name()} + {self.get_printable_var()}'

    def get_inflated_manifold_bounds(self, scale_multiplier):
        bounds = self.get_manifold_bounds()
        offset = scale_multiplier * self.normal_noise_scale
        return bounds[0] - offset, bounds[1] + offset

    def get_full_space_bounds(self):

        start, end = self.get_manifold_bounds()
                # same as for 1e-6
        used_scale = 1e-3 if self.normal_noise_scale is None else self.normal_noise_scale

        #log_scale = np.log10(used_scale ** 2)
        #parts_of_std_dev = 5 if self.dimensions == 2 else 4
        parts_of_std_dev = 5

        x_offset = (parts_of_std_dev+1) * used_scale + 0.25
        x_length = end - start + 2 * x_offset

        y_range = parts_of_std_dev * used_scale
        y_length = 2 * y_range

        x_granularity = (1 / x_length) / 4
        y_granularity = y_length / 2000

        self.y_granularity_of_grid_plot = y_granularity

        x_start, x_end = start - x_offset, end + x_offset

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
        return (x, y), (x_zero, y_zero), (x_zero, y_std_dev_1), (x_zero, -y_std_dev_1)


    @staticmethod
    def arange(start, stop, granularity):
        """this also inserts the last element"""
        x = np.arange(start, stop, granularity)
        return np.insert(x, len(x), stop)

    @staticmethod
    def _kwargs_controller(kwargs):
        only_contains = ['size', 'uniform_noise']
        keys = set(kwargs.keys())
        for key in only_contains:
            if key in keys:
                keys.remove(key)
        assert len(keys) == 0, print(keys)
