

from typing import List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

# Master addition
from master.config import DEVICE
from master.utils.computational import SampleManipulator
from easy_access import EasyAccess
from torch import distributions
#from precomputation.initialization import *
from master.model.real_nvp.model import get_nf_model
#from misc.common_functions import load_config
#from precomputation.loss import Loss
from master.model.real_nvp.flow_loss import get_nf_loss


class ConvNet(nn.Module):

    def __init__(self, args):
        super(ConvNet, self).__init__()

        self.latent_dim = args.latent_dim
        self.num_t = args.num_t
        self.prior = distributions.MultivariateNormal(torch.zeros(self.latent_dim, dtype=torch.float64, device=DEVICE),
                                                 torch.eye(self.latent_dim, dtype=torch.float64, device=DEVICE))
        self.normalization = None
        self.q = None
        self.loss_q = None

        self.load_other_likelihood_model()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.latent_dim)

    def load_other_likelihood_model(self):
        # puts args in easy access
        normal_args = EasyAccess().args
        checkpoint = 'C:/Mega/trunk/master-thesis/data/checkpoints/nf/RealNVP_blocks-6_mnist_cls-0/06.15_14-22-33_normal'
        #nf_args = load_config(checkpoint, 'C:/Mega/trunk/master-thesis/data/')
        nf_args = None
        nf_args.loss_reg = False
        nf_args.new_loss_exp = 1
        #self.q = load_model(checkpoint)
        self.q = get_nf_model(nf_args, nf_args.data_shape)
        #self.loss_q = Loss(nf_args, self.q, nf_args.data_shape)

        self.loss_q = get_nf_loss(nf_args, nf_args.data_shape)

        EasyAccess().args = normal_args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        """
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, z, q_ll, mean=True):
        log_prob = self.prior.log_prob(z)

        # exp(log p_z(f(x)) / exp(log q(x)) = e^(log p_z(f(x)) - log q(x))
        log_prob = log_prob
        weighted_log_prob = log_prob - q_ll
        a = weighted_log_prob.max()

        if self.training:
            normalization = - (a + torch.log(torch.sum( torch.exp(weighted_log_prob - a) ) ) - np.log(len(log_prob)))
        else:
            normalization = self.normalization

        ll = normalization + log_prob
        #ll = - torch.log(torch.sum(torch.exp(log_prob))) + np.log(len(log_prob)) + log_prob
        bpd = ll / (np.log(2) * self.latent_dim)

        loss = - (bpd.mean() if mean else bpd)
        return loss

    def compute_normalization(self, loader):
        weighted_log_probs = []

        self.eval()
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.cuda()
                q_ll = -self.loss_q(self.q, batch) * (np.log(2) * 784)
                z = self.forward(batch)
                log_prob = self.prior.log_prob(z)
                weighted_log_probs.append(log_prob - q_ll)

                if EasyAccess().args.debug and len(weighted_log_probs) > 10: break

            # ignore last batch, might not have the same dim
            weighted_log_probs = torch.stack(weighted_log_probs[:-1]).flatten()

            a = (-weighted_log_probs).max()
            N = weighted_log_probs.shape[0]
            self.normalization = -(a + torch.log(torch.sum(torch.exp(-weighted_log_probs - a))) - np.log(N))
            return self.normalization


    def loss_2(self, z):
        if self.training:
            t = self.prior.sample([self.num_t,])

            log_should_mgf = 0.5 * (t*t).sum(1)

            # repeat for every t
            mz = z.unsqueeze(2).repeat(1, 1, self.num_t)
            # multiply with the t's
            multiplied = mz * t.T
            moment_per_t_and_sample = multiplied.sum(1)
            averaged_per_t = moment_per_t_and_sample.mean(0)

            log_should_mgf - averaged_per_t
            loss = ((log_should_mgf - averaged_per_t)**2).mean()
        else:
            # during evaluation we use the actual log
            ll = self.prior.log_prob(z)
            bpd = ll / (np.log(2) * self.latent_dim)
            loss = - bpd.mean()
            #print(z[0])

        return loss

    def sample(self, num_samples, manipulate_samples=False):
        #return torch.rand((num_samples, 1, 28, 28))
        return self.q.sample(num_samples, manipulate_samples)

    def inverse(self, z):
        #return self.q.inverse(z)
        return torch.rand((len(z), 1, 28, 28))

    @staticmethod
    def loss_creator():
        def loss(model, batch, mean=True):
            z = model.forward(batch)
            ## !!! inserted constant values for dimension
            q_ll = -model.loss_q(model.q, batch) * (np.log(2) * 784)

            bpd = model.loss(z, q_ll, mean)
            return bpd

        return loss




