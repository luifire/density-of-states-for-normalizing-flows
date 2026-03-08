import torch
from torch import distributions
import numpy as np
from torch.utils.data import DataLoader

from master.config import *

# this class sets the first 8 elements of each sample like:
# all 0, all 0.5, * 10, * 100
# 4 x constant
class SampleManipulator:

    def __init__(self):
        self.constant_sample = None

    def manipulate_samples(self, z):
        assert len(z) >= 8
        # just take the first 4 samples to always show them
        if self.constant_sample is None:
            self.constant_sample = z[:4].clone()

        z[0] = torch.zeros_like(z[0])
        z[1] = torch.ones_like(z[0]) * 0.5
        z[2] = z[2] + 1
        z[3] = z[3] + 2
        z[4:8] = self.constant_sample


def evaluate_loader(model, loss_fn, loader, args, inflater=None, pre_computation=None):
    """evaluates an entire loader with the given loss function"""
    # evaluation
    loss_sum = 0.0
    element_count = 0
    logdets = []

    for i, x in enumerate(loader):
        x = x.to(DEVICE)

        # mainly nothing or e.g. iLogits
        if pre_computation:
            x, logdet = pre_computation(x)

        if inflater is not None:
            x = inflater(x)

        loss = loss_fn(model, x, logdet=logdet)
        loss_sum += loss.detach().cpu().item() * len(x)
        element_count += len(x)
        logdets.append(model.logdet().detach())

        if args.debug and i > 10: break

    # log 1/N sum exp( logdet_i ) = log ( sum exp(logdet_i - N ) )
    logdets = torch.cat(logdets)
    avg_logdet = torch.logsumexp(logdets - np.log(element_count), dim=0)

    return loss_sum / element_count, element_count, avg_logdet


def evaluate_data_return_llh(model, loss_fn, data, args):
    batch_size = 2 ** 15 if args.dimensions == 2 else 2 ** 11
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    bpds = []
    for batch in loader:
        bpd = loss_fn(model, batch.cuda(), mean=False)
        bpds.append(bpd.detach().cpu())

    bpds = torch.cat(bpds)
    # remove bpd norming
    llh = -bpds * np.log(2) * args.dimensions
    return llh.numpy()


def create_normal_distribution(data_shape, variance=1):
    assert variance > 0

    """creates the prior functionality with torch"""
    D = int(np.prod(data_shape))

    prior = distributions.MultivariateNormal(torch.zeros(D).to(DEVICE),
                                             variance * torch.eye(D).to(DEVICE))

    return prior