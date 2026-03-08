import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

from warnings import warn

# Master addition
from master.config import DEVICE
from easy_access import EasyAccess
from torch import distributions
from flow_ssl.realnvp.coupling_layer import Rescale
from flow_ssl.realnvp.scale_layer import ScaleLayer


def to_normal(X):
    return float(X.detach().cpu())


class TwoDNet(nn.Module):

    def __init__(self, coupling_layer_count, nice_or_realnvp, args):
        """coupling_layer_count (will NOT be multiplied by 2)"""
        super(TwoDNet, self).__init__()

        self._log_det = None
        self.prior = MultivariateNormal(torch.zeros(2, dtype=torch.float).cuda(),
                                        torch.eye(2, dtype=torch.float).cuda())
        self.nice_implementation = nice_or_realnvp == 'NICE'

        st_nets = []
        rescalings = []
        for _ in range(coupling_layer_count):
            st_net = nn.Sequential(
                #nn.Linear(1, 2))
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )
            st_nets.append(st_net)
            # Learnable scale for s / adapted from original code
            rescalings.append(Rescale(1, one_dimensional=True))

        self.store_for_fullys = nn.Sequential(*st_nets)
        self.store_for_resacles = nn.Sequential(*rescalings)

        self.st_nets = st_nets
        self.rescalings = rescalings
        self.new_loss = 'new_loss' in dir(args) and args.new_loss

        if self.nice_implementation:
            self.scale_layer = ScaleLayer((2), args)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        X = batch[..., 0].squeeze(1)
        Y = batch[..., 1].squeeze(1)
        log_det = torch.zeros(len(batch)).cuda()
        for i, (st_net, rescaling) in enumerate(zip(self.st_nets, self.rescalings)):
            st = st_net(X)
            Y = Y.squeeze(1)

            s = st[..., 0]
            t = st[..., 1]

            if self.nice_implementation:
                Y = Y + t
            else:  # RealNVP
                # adapted from original code
                s = rescaling(torch.tanh(s))
                Y = (Y + t) * s.exp()
                log_det += s

            Y = Y.unsqueeze(1)

            # swap
            tmp = Y
            Y = X
            X = tmp

        # stack invertetly
        batch = torch.stack((Y, X), axis=2)
        # final scaling layer for NICE
        if self.nice_implementation:
            batch = self.scale_layer(batch)
            log_det += self.scale_layer.logdet()

        self._log_det = log_det
        return batch

    def log_prob(self, x):
        z = self.forward(x)
        prior = self.prior.log_prob(z)
        return prior.squeeze(1) + self.logdet()

    def loss(self, x, mean=True, logdet=None):
        log_prob = self.log_prob(x)
        if logdet is not None:
            log_prob += logdet
        nll = -log_prob
        bpd = nll / np.log(4)  # 2*log(2) = log(4)
        if mean:
            bpd = bpd.mean()

        return bpd

    def full_information_loss(self, x):
        z = self.forward(x)
        prior = self.prior.log_prob(z)
        llh = prior.squeeze(1) + self.logdet()
        bpd = -llh / np.log(4)
        return bpd, z, self.logdet()

    def sample(self, num_samples, manipulate_samples=False):
        z = self.prior.sample((num_samples,))
        return self.inverse(z)

    def inverse(self, batch):
        assert len(self.st_nets) % 2 == 0  # otherwise we need to flip X and Y

        log_det = torch.zeros(len(batch)).cuda()

        # final scaling layer for NICE
        if self.nice_implementation:
            batch = self.scale_layer.inverse(batch)
            log_det += self.scale_layer.logdet()

        X = batch[:, 0]
        Y = batch[:, 1]
        for i, (st_net, rescaling) in enumerate(zip(self.st_nets[::-1], self.rescalings[::-1])):
            if i == 9:
                i = i
            st = st_net(X.unsqueeze(1))
            #Y = Y.squeeze(1)

            s = st[..., 0]
            t = st[..., 1]

            if self.nice_implementation:
                Y = Y - t
            else:  # RealNVP
                # adapted from original code
                s = rescaling(torch.tanh(s))
                inv_exp_s = s.mul(-1).exp()
                Y = Y * inv_exp_s - t
                log_det += -s

            # swap
            tmp = Y
            Y = X
            X = tmp

        batch = torch.stack((Y, X), axis=1)
        self._log_det = log_det
        return batch

    def sample_log_prob(self, num_samples):
        z = self.prior.sample((num_samples,))
        z_log_prob = self.prior.log_prob(z)

        if self.nice_implementation:
            # should be positive
            log_det = torch.sum(self.scale_layer.get_scaling_diag())
            x_log_prob = z_log_prob + log_det
            ### test
            #self.inverse(z)  # compute the scalings
            #assert(torch.isclose(-self.logdet(), log_det).all())
        else: # RealNVP
            x = self.inverse(z)  # compute the scalings
            # is negative as the inverse operation stores negative logdet
            log_det = - self.logdet()
            x_log_prob = z_log_prob + log_det

            ### test
            #bpd, z_2, forward_logdet = self.full_information_loss(x.unsqueeze(1).unsqueeze(1))
            #assert torch.isclose(z, z_2.squeeze(1), atol=1e-3).all()
            #assert torch.isclose(forward_logdet, log_det, atol=1e-6).all()

        return x_log_prob, z_log_prob, log_det

    def logdet(self):
        if not self.new_loss or self.nice_implementation:
            return self._log_det
        else:
            avg_log_volume = -torch.logsumexp(-self._log_det, dim=-1)
            return avg_log_volume * torch.ones_like(self._log_det)

    @staticmethod
    def loss_creator():
        def loss(model, batch, mean=True, logdet=None):
            bpd_loss = model.loss(batch, mean, logdet)
            return bpd_loss

        return loss
