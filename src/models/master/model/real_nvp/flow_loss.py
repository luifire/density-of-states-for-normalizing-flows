"""This module is adapted from flow_ssl"""

import numpy as np
import math
import torch.nn as nn
import torch
from warnings import warn

from master.utils.computational import create_normal_distribution
from master.utils.multivariate_student_t import SimpleStudentT
from master.config import *
from master.model.real_nvp.mighty_prior import MightyPrior
from easy_access import EasyAccess


def get_nf_loss(args, data_shape, states, mighty_prior=None):
    if args.flow == 'RealNVPTabular':
        assert False, 'not supported loss'

    if 'student_t' in dir(args) and args.student_t:
        prior = SimpleStudentT(args.deg_freedom, int(np.prod(data_shape)))
    elif mighty_prior is None:
        if 'adjust_prior_var' in dir(args) and args.adjust_prior_var:
            variance = args.var
        else:
            variance = 1
        prior = create_normal_distribution(data_shape, variance)
    else:
        prior = mighty_prior

    flow_loss = FlowLossMaster(prior, k=states)

    return flow_loss


class FlowLossMaster(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior, k=256):
        super().__init__()
        args = EasyAccess().args
        self.counter = 0
        self.k = k
        self.prior = prior

        self.avg_log_volume = None
        self.new_loss = args.new_loss
        self.new_loss_exp = args.new_loss_exp
        self.discard_near_0_z = args.discard_near_0_z if 'discard_near_0_z' in dir(args) else None

        self.use_mighty_prior = 'mighty_prior' in dir(args) and args.mighty_prior
        if isinstance(prior, MightyPrior):
            self.mighty_prior = prior
        elif self.use_mighty_prior:
            self.mighty_prior = MightyPrior(args)
            self.mighty_prior.to(DEVICE)

    def forward(self, model, x, preserve_volume=True, sample_volume=None, mean=True, logdet=None):
        assert preserve_volume

        z = model(x)
        sldj = model.logdet()
        """if self.counter % 1000 == 0:
            scale_logdet = model.scale_layer[0].logdet()[0]
            max_logdet, min_logdet, mean_logdet = sldj.max() - scale_logdet, sldj.min() - scale_logdet, sldj.mean() - scale_logdet
            print("Logdet: ", max_logdet.detach().cpu().numpy(), min_logdet.detach().cpu().numpy(),
                  mean_logdet.detach().cpu().numpy())
        self.counter += 1"""

        if logdet is not None:
            sldj = sldj + logdet
        #if preserve_volume:
        #    sldj = model.logdet()
        #else:
        #    sldj = 0

        if model.training:
            # discard all z for |z| < discard_near_0_z
            if self.discard_near_0_z:
                with torch.no_grad():
                    remover = z.abs() < self.discard_near_0_z
                    keeper = remover.any(dim=1).logical_not()
                    if keeper.logical_not().any():
                        #print('Removed samples from Grad Descent ' + str(int(keeper.logical_not().sum())))
                        new_z = z[keeper]
                        new_sldj = sldj[keeper]

                        z = new_z
                        sldj = new_sldj

                        if len(new_z) == 0:
                            # make sure you don't descent in this case
                            warn('All values close to 0. Will not descent!')
                            return None

        loss = self.loss(z, model, sldj=sldj, sample_volume=sample_volume, mean=mean)

        return loss

    def loss(self, z, model, sldj, mean, y=None, sample_volume=None):
        z = z.reshape((z.shape[0], -1))

        # this comes from normalizing the image to [0, 1]
        # that is img.flatten() / ([k] * size).
        # This is equivalent of a matrix multiplication with 1/k on its diagonal.
        # the determinant if its Jacobian is k^size => log(k) * size
        sldj += -(np.log(self.k) * z.shape[1])

        if self.use_mighty_prior:
            return self.mighty_prior(z, sldj, mean)

        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)

        # the new regularisation should only be used during training
        if self.new_loss:
            if model.training:
                a = (-self.new_loss_exp * sldj).max()
                N = z.shape[0]
                # Vol in original space is logged
                # log sum exp with the maximal a
                # exp * (a + log (1/N sum_i exp( -V(x_i) - a ) ))
                avg_log_volume = -self.new_loss_exp * (a + torch.log(torch.sum(torch.exp(-self.new_loss_exp*sldj - a))) - math.log(N))
            else:
                avg_log_volume = model.avg_log_volume

            ll = prior_ll + avg_log_volume
        else:  # standard loss
            # add volume change
            ll = prior_ll + sldj

        # bpd
        bpd = ll / (math.log(2) * z.shape[1])
        # create loss
        loss = -bpd.mean() if mean else -bpd

        return loss
