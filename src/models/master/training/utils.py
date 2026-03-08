import torch
import numpy as np
import wandb
from warnings import warn
import time
from tqdm import tqdm
from os import path
import platform

# Optim
#from survae.experiments.image.optim.expdecay import get_survae_optim
from master.model.real_nvp.optimizer import get_nf_optimizer
# Loss
from master.model.real_nvp.flow_loss import get_nf_loss
from survae.model.utils.loss import elbo_bpd, loglik_bpd_per_sample
# Model
#from master.model.survae.model_master import get_survae_model
from master.model.real_nvp.model import get_nf_model
from master.data.data_master import simple_data_choice
from iwae.misc import *
from demun.misc import *
from conv_net.conv_net import ConvNet
from two_d_nf.two_d_net import TwoDNet
from res_flow.resflow import get_residual_flow_by_args, get_res_flow_optimizer
from res_flow.compute_loss import compute_loss

#Rest
from master.config import *
from easy_access import EasyAccess


def get_optimizer(args, model, loss=None):
    scheduler_epoch = None
    scheduler_iter = None

    # ugly hack to tell optimizer we want optimization here
    if 'mighty_prior' in dir(args) and args.mighty_prior:
        model.mighty_prior = loss.mighty_prior

    if args.model == 'survae':
        #optimizer, scheduler_iter, scheduler_epoch = get_survae_optim(args, model)
        ...
    elif args.model == 'nf':
        optimizer = get_nf_optimizer(args, model)
    elif args.model == 'iwae':
        optimizer = get_iwae_optimizer(args, model)
    elif args.model in ['demun', 'conv']:
        optimizer = get_demun_optimizer(args, model)
    elif args.model == '2d_nf':
        optimizer = get_nf_optimizer(args, model)
        #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.model == 'res_flow':
        optimizer = get_res_flow_optimizer(args, model)

    return optimizer, scheduler_iter, scheduler_epoch


def get_loss_fn(args, data_shape):
    if args.model == 'survae':
        return elbo_bpd
    elif args.model == 'nf':
        states = 1 if args.dataset in simple_data_choice else 256
        return get_nf_loss(args, data_shape, states=states)
    elif args.model == 'iwae':
        return get_iwae_loss(args, mean=True)
    elif args.model == 'demun':
        return get_demun_loss(args, mean=True)
    elif args.model == 'conv':
        return ConvNet.loss_creator()
    elif args.model == '2d_nf':
        return TwoDNet.loss_creator()
    elif args.model == 'res_flow':
        return compute_loss
    else:
        raise Exception('nope')


def initialize_model(model, args):
    # some model need a forward pass to be initialized
    model.train()
    with torch.no_grad():
        model(torch.zeros((2, *args.data_shape)).cuda())


def recursive_attr_getter(obj, url, skip_last=True):
    attrs = url.split('.')
    for i, attr in enumerate(attrs):
        if skip_last and i+1 == len(attrs):
            break
        obj = obj.__getattr__(attr)
    return obj


def get_model(args, data_shape):
    if args.model == 'survae':
        #model = get_survae_model(args, data_shape=data_shape)
        #model_id = f'survae_steps-{args.num_steps}_scales-{args.num_scales}_pool-{args.pooling}'
        ...
    elif args.model == 'nf':
        model = get_nf_model(args, data_shape)
        model_id = f'{args.flow}_blocks-{args.num_blocks}'
    elif args.model == 'iwae':
        model = get_iwae_model(args, data_shape)
        model_id = f'k-{args.num_k}_latent-{args.latent_dim}_kld-{args.kld_weight}'
    elif args.model == 'demun':
        model = get_demun_model(args, data_shape)
        model_id = f'demun'
    elif args.model == 'conv':
        model = ConvNet(args)
        model_id = 'conv_net'
    elif args.model == '2d_nf':
        model = TwoDNet(args.num_couplings, args.flow, args)
        model_id = f'{args.flow}_2d'
    elif args.model == 'res_flow':
        model = get_residual_flow_by_args(args, (1, *data_shape))
        model_id = 'res_flow'

    print("Model contains {} parameters".format(sum([p.numel() for p in model.parameters()])))

    if args.no_bias:
        warn('turned off bias!!')
        for name, param in model.named_parameters():
            if name.endswith('bias'):
                module = recursive_attr_getter(model, name, skip_last=True)
                module.bias = torch.nn.Parameter(torch.zeros_like(module.bias), requires_grad=False)

    model = model.to(DEVICE)

    model_id += f'_{args.dataset}'
    if not args.inflation:
        model_id += f'_cls-{args.training_class}'

    args.training_id = model_id
    return model, model_id


def get_logging(args, mock_wandb=False, project_name=None):
    """Initializes WANDB"""

    if mock_wandb:
        wandb.log = lambda x, **kwargs: x == x
        wand_run = wandb
        warn('WandB is not Logging.')
    else:
        # always push to debug in case we have debug
        if args.debug:
            project_name = 'ood-debug'
        else:
            #project_name = 'inflation_final'
            project_name = 'inflation'

        wandb_name = time.strftime(".%m.%d %H:%M:%S")
        if 'flow' in dir(args):
            wandb_name += f' {args.flow}'

        training_class = ' t:' + str(args.training_class) if args.training_class != -1 else ''
        wandb_name += ' ' + training_class + f'{args.dataset}'

        if args.inflation:
            if args.var:
                wandb_name += f' var:1e{np.log10(args.var):.2} '
            elif args.uniform_noise:
                wandb_name += f' uniform var:1e{np.log10(args.uniform_noise):.2} '
            else:
                wandb_name += f' var=-inf '

            wandb_name += f'- dim:{args.dimensions}'

        wandb_name += f' {args.wandb_name_ext}'

        settings = None
        if platform.system() == 'Linux':
            settings = wandb.Settings(start_method='fork')
            wandb_name = 'cluster ' + wandb_name

        wand_run = wandb.init(config=vars(args), project=project_name, name=wandb_name, dir=args.log_path,
                              settings=settings)

        args.wandb_name = wandb_name

    EasyAccess().wandb = wand_run
    return wand_run


def apply_model_to_loader(model, loader, loss_fn, args):
    if args.use_new_loss:
        model.compute_avg_log_volume(loader)

    losses_per_sample = []
    for i, (batch, targets) in tqdm(enumerate(loader)):
        batch = batch.to(DEVICE)
        targets = targets.to(DEVICE)

        loss = loss_fn(model, batch)
        losses_per_sample.append(torch.stack((targets, loss)))
        if args.debug and i > 15: break

    return losses_per_sample


def compute_loss_per_class(loss, targets, unique_class_lables):
    """Loss: [a, b, c], targets: [0, 2, 0], all classes : [0, 1, 2]
    returns [a+c, 0, b], [2, 0, 1]
    This adds the losses for the classes individually
    """
    amount_of_unique_classes = len(unique_class_lables)
    # expands the losses, s.t. there is the same loss vector for each class, same for targets
    # M = class x loss
    repeated_loss = loss.repeat(amount_of_unique_classes).view(amount_of_unique_classes, -1)
    # T = class x targets
    repeated_targets = targets.repeat(amount_of_unique_classes).reshape(amount_of_unique_classes, -1)
    # is true where target is equal to cls for T[cls]
    indexed_for_each_class = repeated_targets == unique_class_lables.unsqueeze(1)
    # now M[cls] only has values where there is a loss belonging to cls
    loss_for_each_class = indexed_for_each_class * repeated_loss
    return loss_for_each_class.sum(1), indexed_for_each_class.sum(1)
