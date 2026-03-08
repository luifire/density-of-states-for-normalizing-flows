# Created by David

from torch import nn
import torch
import numpy as np


class ScaleLayer(nn.Module):
    """David: last layer of NICE - adds a scaling for each random variable
    """

    def __init__(self, img_shape, args):
        super(ScaleLayer, self).__init__()

        self.trainable_part = None
        self.fixed_part = None
        self._logdet = None

        input_size = np.prod(img_shape)

        adjust_scaling_weights = 'adjust_scaling_weights' in dir(args) and args.adjust_scaling_weights

        # how many parts can are fixed
        fixed_scaling_size = 0
        logged_scale = None
        if 'trainable_scaling' in dir(args) and args.trainable_scaling is not None \
                and args.normal_scale != None and args.normal_scale > 0:
            fixed_scaling_size = input_size - args.trainable_scaling
            logged_scale = np.log(1 / args.normal_scale)

        # how much is free
        freely_trainable_size = input_size - fixed_scaling_size

        if freely_trainable_size > 0:
            # adjust initialize freely trained scaling parts
            if adjust_scaling_weights and logged_scale is not None:
                trainable_weights = logged_scale * torch.ones(freely_trainable_size)
            else:
                trainable_weights = torch.nn.init.normal_(torch.ones(freely_trainable_size), mean=0.0, std=0.1)

            self.trainable_part = nn.Parameter(trainable_weights.cuda())
        else:
            # otherwise Pytorch complains that there is nothing to train
            self.dummy = nn.Parameter(torch.ones(1))

        # fixed part
        if fixed_scaling_size > 0:
            self.fixed_part = logged_scale * torch.ones(fixed_scaling_size).cuda()

        self.eval_flip = False

    def get_scaling_diag(self):
        if self.trainable_part is not None and self.fixed_part is not None:
            return torch.cat((self.trainable_part, self.fixed_part))
        elif self.trainable_part is not None:
            return self.trainable_part
        else:
            return self.fixed_part

    def forward(self, x, sldj=None, reverse=True):
        self._print_weights()
        scaling_diag_log = self.get_scaling_diag()

        scaling_diag = scaling_diag_log.exp()
        if not torch.isfinite(scaling_diag).all():
            raise RuntimeError('Scaling has NaN or Inf entries' + str(scaling_diag))

        z = torch.matmul(x, torch.diag(scaling_diag))

        self._logdet = torch.sum(scaling_diag_log).repeat(x.shape[0])
        return z

    def inverse(self, z):
        scaling_diag = self.get_scaling_diag()
        x = torch.matmul(z, torch.diag(torch.exp(-scaling_diag)))

        self._logdet = -torch.sum(scaling_diag).repeat(z.shape[0])
        return x

    def logdet(self):
        return self._logdet

    def _print_weights(self):
        # flipflop to print weights
        if self.training is False and self.eval_flip is False:
            self.eval_flip = True
            print('Scaling Diag')
            with torch.no_grad():
                #print(self.get_scaling_diag().exp().detach().cpu())
                scale_diag = self.get_scaling_diag().exp().detach().cpu()
                print(f'Mean: {scale_diag.mean()}, Var: {scale_diag.var()}, '
                      f'Min: {scale_diag.min()}, Max: {scale_diag.max()}')


        if self.training and self.eval_flip:
            self.eval_flip = False