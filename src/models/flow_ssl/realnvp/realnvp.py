import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from flow_ssl.realnvp.coupling_layer import CouplingLayer
from flow_ssl.realnvp.coupling_layer import CouplingLayerTabular
from flow_ssl.realnvp.coupling_layer import MaskCheckerboard, MaskChannelwise, MaskTabular
from flow_ssl.realnvp.coupling_layer import MaskHorizontal, MaskVertical
from flow_ssl.realnvp.coupling_layer import MaskQuadrant, MaskSubQuadrant
from flow_ssl.realnvp.coupling_layer import MaskCenter

from flow_ssl.invertible import iSequential
from flow_ssl.invertible.downsample import iLogits, EvenSpreadOut
from flow_ssl.invertible.downsample import keepChannels
from flow_ssl.invertible.downsample import SqueezeLayer, iAvgPool2d
from flow_ssl.invertible.parts import addZslot
from flow_ssl.invertible.parts import FlatJoin
from flow_ssl.invertible.parts import passThrough

from master.utils.computational import SampleManipulator
from easy_access import EasyAccess
from flow_ssl.realnvp.scale_layer import ScaleLayer
from master.data.data_helper import random_state_keeper
from master.utils.utils import beep


class RealNVPBase(nn.Module):

    def __init__(self, implementation):
        super(RealNVPBase, self).__init__()
        self.implementation = implementation
        self.constant_sample = None
        self.sample_manipulator = SampleManipulator()
        self.scale_layer = None

    def forward(self,x):
        return self.body(x)

    def logdet(self):
        return self.body.logdet()

    def inverse(self, z):
        return self.body.inverse(z)

    def sample(self, num_samples, manipulate_samples=False):
        # Warning: .eval should not happen in case we train the net in an inverse order!

        if 'mighty_prior' in dir(self):  # this is the reference to the mighty prior
            z = self.mighty_prior.sample(num_samples)
        else:
            z = self.prior.sample((num_samples,))
        # tweak outliers see SampleManipulator
        if manipulate_samples:
            self.sample_manipulator.manipulate_samples(z)
        img = self.inverse(z).detach()
        return img

    def sample_with_log_prob(self, num_samples, random_state=None):
        if random_state is not None:
            with random_state_keeper(random_state):
                z = self.prior.sample((num_samples,))
        else:
            z = self.prior.sample((num_samples,))

        samples = self.inverse(z)

        z_log_prob = self.prior.log_prob(z)
        # logdet into x direct, thus we need to invert it
        x_log_prob = z_log_prob - self.logdet()
        return samples, z, x_log_prob, z_log_prob

    def sample_log_prob(self, num_samples):
        """Samples but we are only interested in the log prob.
        This is because it is easy to compute for NICE"""
        z = self.prior.sample((num_samples,))
        z_log_prob = self.prior.log_prob(z)

        if self.implementation in ['nice', 'real_vp']:
            # should be positive (controlled: III)
            log_det = torch.sum(self.scale_layer[0].get_scaling_diag())
            x_log_prob = z_log_prob + log_det
            ### test / remove soon!!!!
            #self.inverse(z)  # compute the scalings
            #assert torch.isclose(-self.logdet(), log_det).all()
        else: # RealNVP
            self.inverse(z)  # compute the scalings
            # is negative as the inverse operation stores negative logdet
            log_det = -self.logdet()
            x_log_prob = z_log_prob + log_det

        return x_log_prob, z_log_prob, log_det

    def nll(self,x,y=None,label_weight=1.):
        print('++++++++++ shouldnt happen')
        exit(1)
        z = self(x)
        logdet = self.logdet()
        z = z.reshape((z.shape[0], -1))
        prior_ll = self.prior.log_prob(z, y,label_weight=label_weight)
        nll = -(prior_ll + logdet)
        return nll

    def set_prior(self, prior):
        self.prior = prior


class RealNVP(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
                 st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=100,
                 implementation='rnvp', ilogits=True, skip_scaling=False):
        super(RealNVP, self).__init__(implementation)

        args = EasyAccess().args
        layers = [addZslot()]
        if ilogits:
            layers.append(passThrough(iLogits()))

        # David
        if 'even_spreadout' in dir(args) and args.even_spreadout:
            layers.append(passThrough(EvenSpreadOut()))

        self.output_shapes = []
        self.avg_log_volume = None
        self.img_shape = img_shape

        _, _, img_width = img_shape

        for scale in range(num_scales):
            in_couplings = self._threecouplinglayers(in_channels, mid_channels, num_blocks, MaskCheckerboard,
                init_zeros, st_type, use_batch_norm, img_width, skip, latent_dim, implementation)
            layers.append(passThrough(*in_couplings))

            if scale == num_scales - 1:
                layers.append(passThrough(
                    CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                        init_zeros, st_type, use_batch_norm, img_width, skip, latent_dim, implementation)))
            else:
                if skip_scaling:
                    # normal case
                    in_couplings = self._threecouplinglayers(in_channels, mid_channels, num_blocks, MaskCheckerboard,
                                                             init_zeros, st_type, use_batch_norm, img_width, skip,
                                                             latent_dim, implementation)
                    layers.append(passThrough(*in_couplings))
                else:
                    layers.append(passThrough(SqueezeLayer(2)))
                    img_width = img_width // 2
                    if st_type != 'autoencoder':  # in the autoencoder case we probably want the bottleneck size to be fixed?
                        mid_channels *= 2
                    out_couplings = self._threecouplinglayers(4*in_channels, mid_channels, num_blocks,
                        MaskChannelwise, init_zeros, st_type, use_batch_norm, img_width, skip, latent_dim, implementation)
                    layers.append(passThrough(*out_couplings))
                    layers.append(keepChannels(2*in_channels))
                    in_channels *= 2

        layers.append(FlatJoin())
        # added by david / this is the final scaling of NICE
        if implementation == 'nice' or implementation == 'real_vp':
            args = EasyAccess().args
            if not ('no_scaling_layer' in dir(args) and args.no_scaling_layer):
                self.scale_layer = [ScaleLayer(img_shape, args)] # stored as list as otherwise pytorch tries to load two scaleLayers
                layers.append(self.scale_layer[0])

        self.body = iSequential(*layers)

    def compute_avg_log_volume(self, loader):
        volumes = []

        self.eval()
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.cuda()
                z = self.forward(batch)
                vol = self.logdet() - (math.log(256) * z.shape[1])
                volumes.append(vol)

                if EasyAccess().args.debug and len(volumes) > 10: break

            # ignore last batch, might not have the same dim
            volumes = torch.stack(volumes[:-1]).flatten()

            a = (-volumes).max()
            N = volumes.shape[0]
            self.avg_log_volume = -(a + torch.log(torch.sum(torch.exp(-volumes - a))) - math.log(N))
            return self.avg_log_volume

    @staticmethod
    def _threecouplinglayers(in_channels, mid_channels, num_blocks, mask_class, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_width=28, skip=True, latent_dim=100, nice_implementation=False):
        layers = [
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False),
                              init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm,
                              img_width=img_width, skip=skip, latent_dim=latent_dim, nice_implementation=nice_implementation),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=True),
                              init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm,
                              img_width=img_width, skip=skip, latent_dim=latent_dim, nice_implementation=nice_implementation),
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_class(reverse_mask=False),
                              init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm,
                              img_width=img_width, skip=skip, latent_dim=latent_dim, nice_implementation=nice_implementation)
        ]
        return layers


class RealNVPCycleMask(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPCycleMask, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=0, output_quadrant=1), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=1, output_quadrant=2), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=2, output_quadrant=3), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskQuadrant(input_quadrant=3, output_quadrant=0), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )

class RealNVPSmall(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPSmall, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks,
                    MaskCheckerboard(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks,
                    MaskCheckerboard(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )


class RealNVP8Layers(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVP8Layers, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=False),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskCheckerboard(reverse_mask=True),
                    init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm, img_width=img_shape[2])),
                FlatJoin()
            )


class RealNVPMaskHorizontal(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPMaskHorizontal, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )


class RealNVPMaskHorizontal3Layers(RealNVPBase):

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8, init_zeros=False,
            st_type='resnet', use_batch_norm=True, img_shape=(1, 28, 28), skip=True, latent_dim=None):
        super(RealNVPMaskHorizontal3Layers, self).__init__()

        self.body = iSequential(
                addZslot(),
                passThrough(iLogits()),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=True),  init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                passThrough(CouplingLayer(in_channels, mid_channels, num_blocks, MaskHorizontal(reverse_mask=False), init_zeros=init_zeros, st_type=st_type, use_batch_norm=use_batch_norm)),
                FlatJoin()
            )


class RealNVPTabular(RealNVPBase):

    def __init__(self, in_dim=2, num_coupling_layers=6, hidden_dim=256, 
                 num_layers=2, init_zeros=False, dropout=False):

        super(RealNVPTabular, self).__init__()
        
        self.body = iSequential(*[
                        CouplingLayerTabular(
                            in_dim, hidden_dim, num_layers, MaskTabular(reverse_mask=bool(i%2)), init_zeros=init_zeros, dropout=dropout)
                        for i in range(num_coupling_layers)
                    ])

