"""Copied and adjusted from:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py
"""

from typing import List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

# Master addition
from master.config import DEVICE
from master.utils.computational import SampleManipulator
from easy_access import EasyAccess

class Demun(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 target_shape: tuple,
                 num_samples: int = 16,
                 deg_of_freedom: int = None,
                 **kwargs) -> None:
        super(Demun, self).__init__()

        self.latent_dim = latent_dim
        self.num_samples = num_samples

        if deg_of_freedom is None:
            self.prior = MultivariateNormal(torch.zeros(latent_dim).cuda(), torch.eye(latent_dim).cuda())
        else:
            pass
            #self.prior =
        final_dim_size = np.prod(target_shape)
        self.visible_base_prior = MultivariateNormal(torch.zeros(final_dim_size).cuda(), torch.eye(final_dim_size).cuda())

        self.target_shape = target_shape
        self.final_dim_size = final_dim_size
        self.sample_manipulator = SampleManipulator()

        self.bpd_factor = (torch.tensor(self.target_shape, device=DEVICE).prod() * np.log(2)) ** -1

        # we round later
        #hidden_dims = [latent_dim, latent_dim ** 1.5, latent_dim ** 2, latent_dim ** 2.5,
        #               latent_dim ** 3, latent_dim ** 4]
        #hidden_dims = [latent_dim, latent_dim**2, latent_dim**3]
        hidden_dims = [latent_dim, latent_dim ** 4]

        bias = True
        final_start_dim = int(latent_dim**2.5)
        if EasyAccess().args.debug:
            self.first_linear = nn.Sequential(nn.Linear(latent_dim, final_start_dim),
                                              nn.LeakyReLU())
        else:
            self.first_linear = nn.Sequential(nn.Linear(latent_dim, int(latent_dim**1.5), bias=bias),
                                              nn.LeakyReLU(),
                                              nn.Linear(int(latent_dim**1.5), latent_dim**2, bias=bias),
                                              nn.LeakyReLU(),
                                              nn.Linear(latent_dim**2, final_start_dim, bias=bias),
                                              nn.LeakyReLU())

        self.mu = nn.ModuleList([
                   nn.Linear(final_start_dim, final_dim_size, bias=bias),
                   nn.Tanh()])

        self.inv_cov = nn.ModuleList([nn.Linear(final_start_dim, final_dim_size**2, bias=bias)])  # note that this is quadratic now

    def create_distributions(self, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples

        latent_samples = self.prior.sample((num_samples,)).cuda()

        Z = self.first_linear(latent_samples)
        #Z = Z.view(self.num_samples, 1, self.latent_dim, self.latent_dim)
        #Z = self.deconv(Z)
        #Z = Z.view(self.num_samples, -1)
        # MU
        mu = Z
        for layer in self.mu:
            mu = layer(mu)
        # tanh is between -1 and 1. But we can only have values between 0 and 1.
        mu = (mu + 1) / 2

        # Cov Matrix
        inv_cov = Z
        for layer in self.inv_cov:
            inv_cov = layer(inv_cov)
        inv_cov = inv_cov.view(-1, self.final_dim_size, self.final_dim_size)

        #unique_cov_entries = unique_cov_entries.view(-1, self.target_shape[0], self.target_shape[1], self.target_shape[2])
        unique_inv_cov = torch.triu(inv_cov)

        # volume changing term
        diag = torch.diagonal(unique_inv_cov, dim1=1, dim2=2)

        # make the main variances positive
        unique_inv_cov = unique_inv_cov - torch.diag_embed(diag) + torch.diag_embed(torch.abs(diag))

        # !!! show again that this is correct
        logd_det = (diag.abs() + torch.finfo(torch.float32).eps).log().sum(-1)

        return mu, unique_inv_cov, logd_det, latent_samples

    def loss_function(self, distribution_params, X, **kwargs) -> dict:
        mu = distribution_params[0]
        unique_inv_cov = distribution_params[1]
        logd_det = distribution_params[2]
        samples = distribution_params[3]
        log_weights = self.prior.log_prob(samples)

        B = X.shape[0]
        # flatten without batch dim
        X = X.flatten(1)

        X = X.unsqueeze(0).repeat(self.num_samples, 1, 1)

        repeated_mu = mu.unsqueeze(0).repeat(B, 1, 1).permute(1, 0, 2)
        # unique_inv_cov.permute(1,2,0)?
        #X - repeated_mu
        X = (X - repeated_mu)
        Z = torch.matmul(X, unique_inv_cov)
        #Z = uncorrelated - repeated_mu

        p_z = self.visible_base_prior.log_prob(Z)

        log_prob = p_z + logd_det.unsqueeze(1).repeat(1, B)

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        # David: he uses softmax here, as in the paper there we use weight and not log_weights
        weights = F.softmax(log_weights, dim=-1).detach()

        if self.training:
            loss = -torch.max(log_prob, dim=0)[0]
        else:
            loss = -torch.sum(log_prob * weights.unsqueeze(1).repeat(1, B), dim=0)
        #loss = -torch.max(log_prob * weights.unsqueeze(1).repeat(1, B), dim=0)[0]

        # to bpd (David)
        bpd = loss * self.bpd_factor
        #bpd = loss

        # if per_sample is in kwargs, we don't want the mean
        if 'per_sample' in kwargs and kwargs['per_sample']:
            return {'loss': bpd}
        else:
            return {'loss': bpd.mean()}

    def sample(self, num_samples:int, manipulate_samples=False) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """

        params = self.create_distributions(num_samples)
        mu = params[0]
        unique_inv_cov = params[1]

        mu = mu.detach()
        mu = mu.view(mu.shape[0], *self.target_shape)
        mu = mu[:num_samples]

        return mu
        cov_matrix_batch = torch.linalg.inv(unique_inv_cov)
        #correlated = torch.matmul(mu, cov)

        img = []
        for cov, vals in zip(cov_matrix_batch, mu):
            correlated = torch.matmul(vals, cov)
            img.append(correlated)

        X = torch.stack(img)
        #X = correlated.view(mu.shape[0], *self.target_shape)
        X = X.view(mu.shape[0], *self.target_shape)
        X = X[:num_samples]

        return X
        #if manipulate_samples:
        #    self.sample_manipulator.manipulate_samples(z)

        return torch.rand((32,1,28,28))
        z = torch.randn(num_samples, 1, self.latent_dim).to(DEVICE)

        #if manipulate_samples:
        #    self.sample_manipulator.manipulate_samples(z)

        samples = self.decode(z).squeeze(1)

        # slice correctly
        if self.data_shape == (1, 28, 28):
            samples = samples[:, :, 1:29, 1:29]

        return samples

    def inverse(self, z):
        """Creates the images from z. z == [B, latent_dim]"""
        img = self.decode(z.unsqueeze(1))

        img = img.squeeze(1)
        # slice correction
        if self.data_shape == (1, 28, 28):
            return img[:, :, 1:29, 1:29]

