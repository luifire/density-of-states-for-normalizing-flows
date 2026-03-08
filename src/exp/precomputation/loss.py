import torch
import numpy as np

# Loss
from master.model.real_nvp.flow_loss import get_nf_loss
from conv_net.conv_net import ConvNet
from two_d_nf.two_d_net import TwoDNet
from res_flow.compute_loss import compute_full_res_flow_loss


class Loss:
    """This class contains different loss functions used for evaluation"""

    def __init__(self, args, model, data_shape, states=256, mighty_prior=None):
        """Mighty Prior needs to be loaded in a dumb way (part of model)"""
        self.model = model
        self.model_name = args.model
        self.image_dimension = np.prod(data_shape)

        if self.model_name == 'nf':
            self.nf_loss = get_nf_loss(args, data_shape, states=states, mighty_prior=mighty_prior)

    def __call__(self, x):
        return self.lln(x)

    def lln(self, x):
        """Basic loss as you expect it"""
        self.model.eval()
        with torch.no_grad():
            if self.model_name == 'nf':
                return -self.nf_loss(self.model, x)
            elif self.model_name == 'iwae':
                results = self.model.forward(x)
                return -self.model.loss_function(*results, M_N=1, per_sample=True)['loss']
            elif self.model_name == 'conv':
                lln, _, _ = self.x_to_full_info(x)
                return lln

    def x_to_full_info(self, x):
        """Computes loss (i.e. lln in bpd), z and volume change"""
        if self.model_name == 'survae':
            lln, z, volume = self.model.x_to_full_info(x)
        elif self.model_name == 'nf':
            z = self.model(x)
            volume = self.model.logdet()
            lln = -self.nf_loss.loss(z, self.model, volume, mean=False)

            # adjust in case one dimension of a color pixel can take more than 256 values
            # (they don't included the color_pixel / 256 at the beginning)
            volume += - np.log(self.nf_loss.k) * z.shape[1]
        elif self.model_name == '2d_nf':
            bpd, z, volume = self.model.full_information_loss(x)
            lln = -bpd
        elif self.model_name == 'res_flow':
            bpd, z, volume = compute_full_res_flow_loss(self.model, x)
            lln = -bpd
        elif self.model_name == 'iwae':
            results = self.model.forward(x)
            losses = self.model.loss_function(*results, M_N=1, per_sample=True)
            lln = -losses['loss']
            # the mean has no use, we could use the first entry. But it feels better ^^
            z = mu = results[2].mean(1)
            volume = losses['KLD']
        elif self.model_name == 'conv':
            z = self.model.forward(x)
            q_ll = self.model.loss_q(self.model.q, x)
            bpd = self.model.loss(z, q_ll, mean=False)
            volume = q_ll
            lln = -bpd

        return lln, z, volume

    def lln_detached(self, batch):
        return self.lln(batch.cuda()).detach().cpu().numpy()
