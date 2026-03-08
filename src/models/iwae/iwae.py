"""Copied and adjusted from:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py
"""

from typing import List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

from iwae.base import BaseVAE

# Master addition
from master.config import DEVICE
from master.utils.computational import SampleManipulator


class IWAE(BaseVAE):

    def __init__(self,
                 latent_dim: int,
                 data_shape: tuple,
                 num_samples: int = 5,
                 kld_train_weight=1,
                 depth=5,
                 **kwargs) -> None:
        super(IWAE, self).__init__()

        self.pre_flattening_dim = None
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.kld_train_weight = kld_train_weight
        self.sample_manipulator = SampleManipulator()

        self.bpd_factor = (torch.tensor(self.data_shape, device=DEVICE).prod() * np.log(2)) ** -1

        use_bias = True
        if use_bias is False:
            print('no bias for iwae?')
            exit(1)
        modules = []
        hidden_dims = [32, 64, 128, 256, 512, 1024]
        hidden_dims = hidden_dims[:depth]

        in_channels = data_shape[0]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            #print(f'{in_channels} -> {h_dim}')
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        if data_shape == (1, 28, 28):
            depth_to_bottle_neck = {6: 1024, 5: 512, 4: 1024, 3: 2048}#, 2:3136, 1:6272}
            before_bottle_neck_dim = depth_to_bottle_neck[depth]

        #print(f'{before_bottle_neck_dim} -> {latent_dim}')
        self.fc_mu = nn.Linear(before_bottle_neck_dim, latent_dim, bias=use_bias)
        self.fc_var = nn.Linear(before_bottle_neck_dim, latent_dim, bias=use_bias)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, before_bottle_neck_dim, bias=use_bias)
        #print(f'{latent_dim} -> {before_bottle_neck_dim}')

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
            #print(f'{hidden_dims[i]} -> {hidden_dims[i+1]}')

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=data_shape[0], kernel_size=3, padding=0, bias=use_bias),
                            nn.Tanh())
        #print(f'{hidden_dims[-1]} -> {hidden_dims[-1]}')
        #print(f'{hidden_dims[-1]} -> {data_shape[0]}')

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        self.pre_flattening_dim = result.shape
        result = torch.flatten(result, start_dim=1)

        #print(result.shape)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (Tensor) [B x S x D]
        :return: (Tensor) [B x S x C x H x W]
        """
        B, _, _ = z.size()
        #z = z.reshape(-1, self.latent_dim) #[BS x D]
        z = z.contiguous().view(-1, self.latent_dim)  # [BS x D]
        result = self.decoder_input(z)
        result = result.view(-1, self.pre_flattening_dim[1], self.pre_flattening_dim[2], self.pre_flattening_dim[3])
        result = self.decoder(result)
        result = self.final_layer(result)  # [BS x C x H x W ]
        result = result.view([B, -1, result.size(1), result.size(2), result.size(3)])  # [B x S x C x H x W]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, eps

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z, eps = self.reparameterize(mu, log_var)  # [B x S x D]
        return [self.decode(z), input, mu, log_var, z, eps]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        eps = args[5]

        input = input.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)  # [B x S x C x H x W]

        # David 0.1 as suggested by te WAIC paper
        if self.training:
            kld_weight = self.kld_train_weight
        else:
            kld_weight = 1
        # slice correctly
        if self.data_shape == (1, 28, 28):
            recons = recons[:, :, :, 1:29, 1:29]

        #print(recons.shape)
        log_p_x_z = ((recons - input) ** 2).flatten(2).mean(-1)  # Reconstruction Loss [B x S]
        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=2)  # [B x S]
        # Get importance weights
        log_weight = log_p_x_z + kld_weight * kld_loss

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        # David: he uses softmax here, as in the paper there we use weight and not log_weights
        weight = F.softmax(log_weight, dim=-1).detach()

        loss = torch.sum(weight * log_weight, dim=-1)

        # to bpd (David)
        bpd = loss * self.bpd_factor

        # if per_sample is in kwargs, we don't want the mean
        if 'per_sample' in kwargs and kwargs['per_sample']:
            return {'loss': bpd, 'Reconstruction_Loss': log_p_x_z.mean(1), 'KLD': -kld_loss.mean(1)}
        else:
            return {'loss': bpd.mean(), 'Reconstruction_Loss': log_p_x_z.mean(), 'KLD': -kld_loss.mean()}

    def sample(self, num_samples:int, manipulate_samples=False) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1, self.latent_dim).to(DEVICE)

        if manipulate_samples:
            self.sample_manipulator.manipulate_samples(z)

        samples = self.decode(z).squeeze(1)

        # slice correctly
        if self.data_shape == (1, 28, 28):
            samples = samples[:, :, 1:29, 1:29]

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        Returns only the first reconstructed sample
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        #return self.forward(x)[0][:, 0, :]

    def inverse(self, z):
        """Creates the images from z. z == [B, latent_dim]"""
        img = self.decode(z.unsqueeze(1))

        img = img.squeeze(1)
        # slice correction
        if self.data_shape == (1, 28, 28):
            return img[:, :, 1:29, 1:29]

