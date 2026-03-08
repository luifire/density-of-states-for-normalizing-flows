import torch
import math
import numpy as np

from easy_access import EasyAccess


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def add_padding(x, args, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def compute_full_res_flow_loss(model, x, beta=1.0):
    args = EasyAccess().args

    # bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(1).to(x)
    # logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    """
    if args.data == 'celeba_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256"""

    nvals = 1
    x, logpu = add_padding(x, args, nvals)

    # if args.squeeze_first:
    #    x = squeeze_layer(x)
    ##
    input_size = x.shape
    ##
    z, delta_logp = model(x.view(-1, *input_size[1:]), 0)

    # log p(z)
    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

    ##
    im_dim = 1
    # args.padding is 0!
    ##

    # log p(x)
    # logpx = logpz - beta * delta_logp - np.log(nvals) * (args.imagesize * args.imagesize * (im_dim + args.padding)) \
    logpx = logpz - beta * delta_logp - np.log(nvals) * (args.dimensions * (im_dim + args.padding)) \
            - logpu
    bits_per_dim = -logpx / (args.dimensions * im_dim) / np.log(2)

    # logpz = torch.mean(logpz).detach()
    # delta_logp = torch.mean(-delta_logp).detach()

    # return bits_per_dim, logits_tensor, logpz, delta_logp

    #bpd, z, volume
    bits_per_dim = bits_per_dim.squeeze(1)
    delta_logp = delta_logp.squeeze(1)
    return bits_per_dim, z, -delta_logp


def compute_loss(model, x, beta=1.0, mean=True):
    bits_per_dim, _, _ = compute_full_res_flow_loss(model, x, beta)

    if mean:
        bits_per_dim = bits_per_dim.mean()

    return bits_per_dim
