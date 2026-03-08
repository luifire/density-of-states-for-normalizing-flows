import unittest
import sys
import os
import wandb
from unittest.mock import patch
from time import time

from master.utils.arguments import load_args
from master.data.data_master import get_data, create_evaluation_loaders
from master.training.training_master import FlowExperiment_Master
from easy_access import EasyAccess
from master.utils.utils import print_line, create_folders
from survae.experiments.image.utils import set_seeds

# Model
from master.training.utils import *

from mocks.mock_optimizer import MockedOptimizer
from mocks.mock_model import MockModel
from mocks.mock_misc import *


class InitializeExperiment:
    """This mocks a model and tests the training environment"""

    @staticmethod
    def initialize():
        InitializeExperiment.data_root = os.environ.get('data_path', '/master-thesis-work/data/')
        InitializeExperiment.log_path = os.environ.get('dummy_path', '/master-thesis-work/data/dummy/')
        InitializeExperiment.intermediate_checks = os.path.join(InitializeExperiment.log_path, 'intermediate_checks')
        InitializeExperiment.training_class = os.environ.get('training_class', 0)

    @staticmethod
    def initialize_training(args, mock_model=True, mock_wandb=True):
        """This initializes all needed modules for training (model, optimizer, ...)
        if wandb_name == None: don't log on wandb"""
        print(args)
        # command line arguments
        args = load_args(args.split(' '))

        #args.debug = True
        # constant seed, also make all functions deterministic
        set_seeds(19901005, cuda_deterministic=True)

        args.log_path = InitializeExperiment.log_path
        args.intermediate_checks = InitializeExperiment.intermediate_checks
        if args.wandb_name_ext is None:
            args.wandb_name_ext = 'unit_test'
            print('No wandb_name_ext set. I use "unit_test".')

        ##############
        # Data Loading (Common)
        args.data_path = os.path.join(args.root, 'datasets')

        train_loader, eval_loader, data_shape = get_data(args, args.training_class)
        args.data_shape = data_shape

        other_classes_loaders = create_evaluation_loaders(args, use_train_loaders=False)

        ##############
        # Model Loading (Split)
        if mock_model:
            model = MockModel()
            model_id = args.training_id = 'MockModel'
        else:
            model, model_id = get_model(args, data_shape)

        ##############
        ## Loss Function (Split)
        if mock_model:
            loss_fn = mock_loss_fn
        else:
            loss_fn = get_loss_fn(args, data_shape)

        ##############
        # Optimizer (Split)
        if mock_model:
            optimizer = MockedOptimizer()
            scheduler_epoch = None
            scheduler_iter = None
        else:
            optimizer, scheduler_iter, scheduler_epoch = get_optimizer(args, model)

        ##############
        # Folder (Common)
        create_folders(args, model_id)

        ##############
        # Initialize W&B (Common)
        wand_run = get_logging(args, mock_wandb, project_name='ood-unit-tests')
        ##############

        ## Training (Common)
        exp = FlowExperiment_Master(args=args,
                                    epochs=args.epochs,
                                    data_id=args.dataset,
                                    model_id=model_id,
                                    train_loader=train_loader,
                                    eval_loader=eval_loader,
                                    other_classes_loaders=other_classes_loaders,
                                    model=model,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    scheduler_iter=scheduler_iter,
                                    scheduler_epoch=scheduler_epoch,
                                    logging=wand_run)
        return exp
