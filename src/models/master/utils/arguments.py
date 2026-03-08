import argparse
import os
import pickle
import numpy as np

from easy_access import EasyAccess

from master.config import *

# SurVAE
from master.data.data_master import add_data_args
#from master.model.survae.model_master import add_model_args
#from survae.experiments.image.optim.expdecay import add_optim_args

# NF
from master.model.real_nvp.arguments import add_nf_args
from master.data.data_master import is_dataset_colored

# IWAE
from iwae.misc import add_iwae_args

# Res Flow
from res_flow.rest.arguments_res_flow import add_res_flow_arguments


def load_args(command_line=None):
    # Parse all arguments
    parse_for_model = argparse.ArgumentParser()
    parse_for_model.add_argument('--model', type=str, choices=['survae', 'nf', 'iwae', 'demun',
                                                               'conv', '2d_nf', 'res_flow'])

    args, _ = parse_for_model.parse_known_args(args=command_line)
    model = args.model.lower()

    # fully load arguments
    parser = argparse.ArgumentParser()

    # Master Extension
    parser.add_argument('--training_class', type=int, default=-1,
                        help='This is the class we will train upon. All others will be removed for training.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str)
    parser.add_argument('--own_folder', default='', type=str, help='special folder, will be created in '
                                                                   '--root/checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None)

    parser.add_argument('--num_samples', default=24, type=int, help='Number of samples at test time')
    parser.add_argument('--sample_every', default=None, type=int, help='when to sample')
    parser.add_argument('--overfit', help='Train on a single batch. Note that you should adjust batch_size.',
                        action='store_true', default=False)
    parser.add_argument('--no_bias', help='Deactivates the bias.', action='store_true', default=False)
    parser.add_argument('--early_stopping', type=str, #, choices=['best_of_30', 'best_of_60', 'None'],
                        help='Stops when loss has not improved for 30 epochs.', default='None')

    parser.add_argument('--wandb_name_ext', default='', type=str, help='Will be added to the wandb run name')
    parser.add_argument('--adjust_prior_var', default=False, action='store_true',
                        help='If true will make the prior distributed according to the variance of the inflation noise')

    parser.add_argument('--manifold_evaluation', type=int, default=None)
    parser.add_argument('--pdf_evaluation', type=int, default=None)
    parser.add_argument('--distance_evaluation', type=int, default=None)

    parser.add_argument('--adjust_scaling_weights', default=False, action='store_true',
                        help='if true sets the scaling weight s.t. they match the std dev of the inflation')
    parser.add_argument('--trainable_scaling', default=None, type=int,
                        help='if true sets the scaling weight s.t. they match the std dev of the inflation'
                             'and makes it none-trainable')
    parser.add_argument('--mighty_prior', default=False, action='store_true', help='creates mighty prior')
    parser.add_argument('--discard_near_0_z', default=None, type=float,
                        help='f(x) near 0 should cause infinite gradients. '
                             'Close to 0 will also be explosive. We will discard x for |f(x)| < discard_near_0_z')

    parser.add_argument('--pre_inflation_ilogits', default=False, action='store_true', help='applys ilogits pre inflation')
    parser.add_argument('--clip_gradient', default=False, action='store_true')
    parser.add_argument('--normal_init_trainable_scale', default=True, action='store_true',
                        help='if true the trainable scale will be initialized normally. Otherwise the adjusted scaling is used')


    add_inflation_arguments(parser)
    add_exp_args(parser)
    add_data_args(parser)
    if model == 'survae':
        #add_model_args(parser)
        #add_optim_args(parser)
        ...
    elif model == 'nf':
        add_nf_args(parser)
    elif model == 'iwae':
        add_iwae_args(parser)
    elif model == 'demun':
        add_iwae_args(parser)
    elif model == 'conv':
        parser.add_argument('--latent_dim', type=int)
        parser.add_argument('--num_t', type=int)
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    elif model == '2d_nf':
        parser.add_argument('--num_couplings', default=10, type=int, help='number of coupling layers.')
        parser.add_argument('--prior', choices=['Gaussian'], default='Gaussian')
        parser.add_argument('--flow', type=str, default='RealNVP', choices=['RealNVP', 'NICE'],
                            help='Flow model to use (default: RealNVP)')

        parser.add_argument('--optim', choices=['Adam', 'RMSprop'], default='Adam')
        parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
        parser.add_argument('--weight_decay', default=5e-5, type=float,
                            help='L2 regularization (only applied to the weight norm scale factors)')
        parser.add_argument('--new_loss', action='store_true', default=False, help='new loss created by David')
    elif model == 'res_flow':
        add_res_flow_arguments(parser)

    ############### Parsing ###################
    args = parser.parse_args(args=command_line)

    if model == 'survae' or model == 'nf':
        args.use_new_loss = args.new_loss
    else:
        args.use_new_loss = False

    if model == 'res_flow':
        args.flow = 'res Flow'
    if 'mmd_reg' not in dir(args):
        args.mmd_reg = False

    args.scale = None
    if args.var is not None:
        args.scale = np.sqrt(args.var)
        print(f'Log_10(var) = {np.log10(args.var)}')

    args.tangent_scale = None
    if args.tangent_noise_var is not None:
        args.tangent_scale = np.sqrt(args.tangent_noise_var)
        print(f'Log_10(tangent var) = {np.log10(args.tangent_noise_var)}')

    if args.tangent_scale is None and args.var is not None:
        args.tangent_scale = args.scale
        args.tangent_noise_var = args.var

    assert not (args.uniform_noise and args.var), 'either normal or uniform noise'
    assert not (args.uniform_noise and args.tangent_noise_var), 'not implemented'

    if args.early_stopping == 'None':
        args.early_stopping = None
    else:
        if args.early_stopping == 'best_of_30':
            args.early_stopping = 30
        else:
            args.early_stopping = int(args.early_stopping)

    if args.debug:
        _set_debug_options(args)

    args.num_bits = 32 if is_dataset_colored(args.dataset) else 8
    args.epochs += 1  # sometimes necessary to make the last evaluation

    EasyAccess().args = args
    return args


def _set_debug_options(args):
    import warnings
    warnings.warn('\n---------------------\n'
                  'Debug is activated\n'
                  '---------------------')
    args.num_workers = 0

    if args.model == 'survae':
        args.densenet_depth = 1
        args.num_steps = 2
    elif args.model == 'nf':
        args.num_blocks = 2


def add_inflation_arguments(parser):
    parser.add_argument('--inflation', action='store_true')
    parser.add_argument('--var', type=float, default=None, help='Sigma^2 / variance of Gauss Distribution that will be added for inflation')
    parser.add_argument('--uniform_noise', type=float, default=None, help='enter the related variance this uniform'
                                                                          'noise should relate to.'
                                                                          'Then the uniform noise will be adjusted to'
                                                                          'this range')
    parser.add_argument('--tangent_noise_var', type=float, default=None,
                        help='Uses "var" in the normal space as noise and "tangent_noise" as variance for the '
                             'tangent space. This is to check, whether we perform better with high noise'
                             'due to more noise in the tangent space.')

    parser.add_argument('--dimensions', type=int, default=-1, help='Dimensionality in which we embed the lower dimensional manifold')
    parser.add_argument('--train_size', type=int, default=-1, help='amount of data in the training set.')
    parser.add_argument('--even_spreadout', default=False, action='store_true', help='-0.5 to data [0,1] -> [-.5, .5]')
    parser.add_argument('--no_rotation', action='store_true', default=False, help='Does not rotate into high dimensions')
    parser.add_argument('--list_dimension', action='store_true', default=False, help='Does not arrange the data like an image')
    parser.add_argument('--skip_scaling', action='store_true', default=False,
                        help='there is some channel wise coupling layer stuff, which will be ignored with this flag.'
                             'This one can keep the scaling parameter while still having the same amount of coupling layers')

    # von Mises on Ring
    parser.add_argument('--sphere_radius', type=float, default=0, help='Sphere: radius of the manifold sphere')
    parser.add_argument('--mises_kappa', type=float, default=6., help='Sphere: von Mises parameter')
    # Uniform dist
    parser.add_argument('--line_length', type=float, default=1., help='Line: length of the latent line')
    # Sigmoid (not used anymore)
    parser.add_argument('--x_start', type=float, default=-15., help='Sigmoid: start of x for sig(x)')
    parser.add_argument('--x_end', type=float, default=20., help='Sigmoid: end of x for sig(x)')
    parser.add_argument('--growth_rate', type=float, default=1., help='Sigmoid: groth rate for the logistic function')
    # Smooth Step
    parser.add_argument('--data_start', type=float, default=-1., help='Sigmoid: start of x for sig(x)')
    parser.add_argument('--data_end', type=float, default=0.75, help='Sigmoid: end of x for sig(x)')
    parser.add_argument('--data_offset', type=float, default=1., help='Sigmoid: for lower part + offset')
    parser.add_argument('--data_range', type=float, default=3., help='Sigmoid: Relation to upper likelihood')
    parser.add_argument('--data_growth', type=float, default=4., help='Sigmoid: Speed of growth')


def add_exp_args(parser):
    """Official experiment arguments.
    I just moved it here """
    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--resume', type=str, default=None)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    #parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--dont_check', help='stops checking at all. Otherwise checks if evaluation increases',
                        action='store_true', default=False)


def save_args(log_path, args):
    model_parameters = os.path.join(log_path, 'model_parameter')
    os.makedirs(model_parameters, exist_ok=True)

    print(f'save model parameter to {model_parameters}')
    with open(model_parameters + 'args', 'wb') as output:
        pickle.dump(args, output, pickle.HIGHEST_PROTOCOL)

