"""This module prepares all the sampling that is shown in the evaluation"""
import torch
import numpy as np
from warnings import warn
from enum import Enum
from survae.experiments.image.utils import set_seeds

# Helper Functions
from shared_functions import print_line
from misc.common_functions import digitize_image_values
from misc.constants import *


class SampleType(Enum):
    """Different types of samples
    If they contain the word "LIKELIHOOD" there we stored (imgs, llns)"""
    RANDOM = 1
    RANDOM_LIKELIHOOD = 2
    DIGITIZED_LIKELIHOOD = 3
    UNIT_CIRCLE_LIKELIHOOD = 4
    RANDOM_FACTOR = 5


class PreSampling:
    CLEANING_RESOLUTION = 16

    def __init__(self, information_container):
        self.storage = information_container

    def run(self):
        print_line()
        print('Sampling...')

        sample_storage = dict()
        # draw a lot of samples
        self._draw_random_samples(sample_count=24, sample_storage=sample_storage)
        self._samples_with_likelihood(sample_storage)
        self._unit_circle_sample(sample_storage)
        self._draw_random_samples_with_factor(sample_storage)

        return sample_storage

    @classmethod
    def _tensor_to_cpu_img(cls, tensor):
        return tensor.detach().float().cpu()

    def _draw_random_samples_with_factor(self, sample_storage):
        print('Sample f*N')
        """This does something like for f in factors plot(sample(f * z)), z~N(0,1) """
        factors = [0.25, 0.5, 0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.5]
        random_samples = 5

        line_of_images = []  # one entry contains all images from that particular sampled z with its factors
        for y in range(random_samples):
            set_seeds(y)
            latent_shape = self.storage.latent_shape
            # ugly, but torch.normal wants it like that and one dimensional arrays don't work and (1) = 1
            init_random = torch.normal(0, 1, size=latent_shape)
            Z = []
            for x, f in enumerate(factors):
                set_seeds(self.storage.config.seed)
                z = f * init_random
                # s.t. torch.cat operates on the first dimension
                z = z.unsqueeze(0)
                Z.append(z)

            Z = torch.cat(Z)
            imgs = self.sample_from_z(Z)
            imgs = self._tensor_to_cpu_img(imgs)
            line_of_images.append(imgs)

        sample_storage[SampleType.RANDOM_FACTOR] = (line_of_images, factors)

    def _unit_circle_sample(self, sample_storage):
        """This samples in latent space with img ~ model(z), where z always has the same value
        like [0, 0.1, 0.2, ..., 2]"""
        print('Sample f*1')
        small = np.arange(0, .5, 0.025)
        big = np.arange(.5, 2, 0.2)
        factors = np.concatenate([small, big]).flatten()

        Z = []
        for i, f in enumerate(factors):
            # there is a lot of sampling going on for SurVAE - thus we reset this for each image
            set_seeds(self.storage.config.seed)
            # draw samples
            z = f * torch.ones(self.storage.latent_shape)
            # so cating works on the correct dimension
            z = z.unsqueeze(0)
            Z.append(z)

        Z = torch.cat(Z)
        imgs = self.sample_from_z(Z)
        # so they can be digitized
        imgs = imgs.type(torch.float)
        # cleaning
        imgs = digitize_image_values(imgs, PreSampling.CLEANING_RESOLUTION)
        # produce likelihood
        llns = self.storage.loss_container.lln_detached(imgs)

        sample_storage[SampleType.UNIT_CIRCLE_LIKELIHOOD] = (self._tensor_to_cpu_img(imgs), llns, factors)

    def _samples_with_likelihood(self, sample_storage):
        """Samples of the main class with likelihood"""
        print('Sample with Likelihood')
        # there is randomness in creating the loss
        set_seeds(self.storage.config.seed)

        samples = self.storage.model.sample(num_samples=12)
        # compute lln
        lln = self.storage.loss_container.lln_detached(samples)
        sample_storage[SampleType.RANDOM_LIKELIHOOD] = (self._tensor_to_cpu_img(samples), lln)

        # digitized images with likelihood
        cleaned_samples = digitize_image_values(samples, original_resolution=PreSampling.CLEANING_RESOLUTION)
        # compute lln
        cleaned_lln = self.storage.loss_container.lln_detached(cleaned_samples)
        sample_storage[SampleType.DIGITIZED_LIKELIHOOD] = (self._tensor_to_cpu_img(cleaned_samples), cleaned_lln)

    def _draw_random_samples(self, sample_count, sample_storage):
        print('Random Samples')
        """This calls the cample function of the model and plots the samples"""
        model = self.storage.model
        samples = model.sample(sample_count)

        sample_storage[SampleType.RANDOM] = self._tensor_to_cpu_img(samples)

    def sample_from_z(self, z):
        self.storage.model.eval()
        with torch.no_grad():
            z = z.to(DEVICE)
            img = self.storage.model.inverse(z)

        return img
