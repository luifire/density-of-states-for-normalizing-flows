from os import path
import pickle
from warnings import warn

import torch
import numpy as np

# Data
from master.data.data_master import get_data, create_evaluation_loaders
# Model
#from master.model.survae.model_master import get_survae_model
from master.model.real_nvp.model import get_nf_model
from iwae.misc import get_iwae_model
from conv_net.conv_net import ConvNet
from two_d_nf.two_d_net import TwoDNet
from res_flow.resflow import get_residual_flow_by_args
from res_flow.master_utils import load_res_flow_model
from master.training.utils import get_loss_fn
from master.utils.utils import get_bool_from_args

# Master Extension Modules
from master.config import *
from easy_access import EasyAccess

# Helper Functions
from misc.constants import *
from precomputation.loss import Loss
from precomputation.pre_evaluation_main import PreEvaluationMain


def create_evaluation_args(args, debug=False):
    """This manipulates the arguments (which are loaded from the training)
    s.t. they can be used for evaluation"""
    EasyAccess().args = args

    if args.model.startswith('nf'):
        args.batch_size = 1024
    elif args.model.startswith('survae'):
        args.batch_size = 128
    elif args.model.startswith('iwae'):
        args.batch_size = 256
    args.eval_batch_size = args.batch_size

    print(f'Batch Size: {args.batch_size}')

    args.augmentation = None
    args.num_workers = 0

    # tweak some settings
    if debug:
        args.batch_count = 1  # this is how many batches we will evaluate
        args.debug = True
        print()
        print(f'maximum amount of images per class: {args.batch_count * args.batch_size}')

    return args


def load_data(args, verbose=True):
    # a dict means we have multiple configs
    # however, they are equivalent when it comes to data loading, so we can just take the first one
    if type(args) is dict:
        args = next(iter(args.values()))

    if verbose:
        print('Datapoints per class')

    train_loader, eval_loader, data_shape = get_data(args, args.training_class, shuffle_train=False, verbose=verbose)

    if args.dataset == 'fashionmnist':
        args.dataset = 'mnist'
        all_loaders = create_evaluation_loaders(args, verbose=verbose)
        args.dataset = 'fashionmnist'
    else:
        all_loaders = create_evaluation_loaders(args, verbose=verbose)
    all_loaders[args.training_class] = eval_loader

    all_loaders[MAIN_TRAIN_CLS] = train_loader
    EasyAccess().data_shape = data_shape

    return all_loaders


def load_preprocessing(checkpoint):
    """This file contains all relevant preprocessed data"""
    preprocessed_file_name = path.join(checkpoint, PreEvaluationMain.FILE_NAME)
    if not path.isfile(preprocessed_file_name):
        raise Exception(f'Preprocessed file {preprocessed_file_name} not found. '
                        f'Run pre_evaluation_main.py')
    preprocessing = pickle.load(open(preprocessed_file_name, 'rb'))

    return preprocessing


def load_preprocessings(checkpoints):
    """Loads all preprocessings from the given checkpoints"""
    llns, samples, artificials, additionals = dict(), dict(), dict(), dict()
    for model, value in checkpoints.items():
        preprocessings = load_preprocessing(value)
        samples[model] = preprocessings['samples']
        artificials[model] = preprocessings['artificial']
        additionals[model] = preprocessings['additional']
        lln = preprocessings['lln']

        """This is transforming into a new data format. 
        I hope this won't be necessary after a certain point"""
        translator = {0: IMG_IDX, 1: PD_IDX, 2: Z_IDX, 3: VOLUME_IDX, 4: T_SCORE, 5: TARGET_IDX}
        if isinstance(lln[MAIN_TRAIN_CLS], list):
            print(value)
            print('------fail')
            for cls in lln:
                new_cls_lln = dict()
                for i, item in enumerate(lln[cls]):
                    new_cls_lln[translator[i]] = item
                lln[cls] = new_cls_lln
        elif isinstance(lln[MAIN_TRAIN_CLS], dict) and 0 in lln[MAIN_TRAIN_CLS]:
            print(model)
            print('------fail')
            for cls in lln:
                new_cls_lln = dict()
                for key, item in lln[cls].items():
                    new_cls_lln[translator[key]] = item
                lln[cls] = new_cls_lln

        llns[model] = lln

    return llns, samples, artificials, additionals


def get_shapes():
    """Returns input data shape, latent shape (of z), the total dimension of the input (w x h x c)"""

    model_type = EasyAccess().args.model
    data_shape = EasyAccess().data_shape
    # Latent Shape
    if model_type == 'survae':
        EasyAccess().latent_shape = EasyAccess().model.latent_shape
    elif model_type == 'nf':
        EasyAccess().latent_shape = [np.prod(data_shape)]
    elif model_type == 'iwae':
        EasyAccess().latent_shape = [EasyAccess().args.latent_dim]
    elif model_type == 'conv':
        EasyAccess().latent_shape = [EasyAccess().args.latent_dim]

    return EasyAccess().data_shape, EasyAccess().latent_shape, np.prod(data_shape)


def load_model(checkpoint, verbose=True, cuda=True):
    args = EasyAccess().args
    # Model
    if args.model == 'survae':
        #model = get_survae_model(args, data_shape=args.data_shape)
        ...
    elif args.model == 'nf':
        model = get_nf_model(args, args.data_shape, verbose=verbose)
    elif args.model == 'iwae':
        model = get_iwae_model(args, args.data_shape)
    elif args.model == 'conv':
        model = ConvNet(args)
    elif args.model == '2d_nf':
        model = TwoDNet(args.num_couplings, args.flow, args)
    elif args.model == 'res_flow':
        model = get_residual_flow_by_args(args, (1, *args.data_shape))

    if get_bool_from_args('mighty_prior'):
        loss_fn = get_loss_fn(args, args.data_shape)
        model.mighty_prior = loss_fn.mighty_prior

    checkpoint = torch.load(checkpoint + '/check/checkpoint.pt')

    # renaming of variables
    scale_diag_label = 'scale_layer.scaling_diag'
    if scale_diag_label in checkpoint['model'].keys():
        val = checkpoint['model'][scale_diag_label]
        del checkpoint['model'][scale_diag_label]
        checkpoint['model']['scale_layer.trainable_part'] = val

    if args.model == 'res_flow':
        load_res_flow_model(model, checkpoint, args)
    else:
        model.load_state_dict(checkpoint['model'])
        if cuda:
            model.to(DEVICE)
        else:
            model.to('cpu')

    EasyAccess().model = model

    return model


def load_loss(random_loader):
    loss_fn = Loss(EasyAccess().args, EasyAccess().model, EasyAccess().data_shape)
    # model needs to have one forward run to be able to sample later
    x, _ = next(iter(random_loader))
    loss_fn(x.to(DEVICE))

    EasyAccess().loss_fn = loss_fn
    return loss_fn
