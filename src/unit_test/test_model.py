import unittest
import sys
import os
import wandb

import torch
from mocks.initialize_mocked_experiment import InitializeExperiment
from master.config import *
from easy_access import EasyAccess

class TestModels(unittest.TestCase):
    overfit_epochs = 100
    big_model_epochs = 50

    small_model_epochs = big_model_epochs
    hierarchical_survae_epochs = big_model_epochs
    eval_every = 1
    #old case: eval_every = 10

    """Here we test some models
    These tests take a while"""
    def setUp(self):
        EasyAccess().reset()
        InitializeExperiment.initialize()
        # some speed up, as these are big tests
        torch.backends.cudnn.benchmark = True

    ###############
    # Overfit Batch
    ###############
    def test_nf_overfit_batch(self):
        args = f'--model nf --root {InitializeExperiment.data_root} --flow=RealNVP --dataset=mnist ' \
               f'--epochs={TestModels.overfit_epochs} --lr=5e-5 ' \
               f'--num_blocks=3 --num_scales 2 --batch_size=2 ' \
               f'--num_workers=0 --training_class {InitializeExperiment.training_class} ' \
               f'--overfit --wandb_name_ext overfit'
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)
        TestModels._deactivate_evaluations(exp)

        exp.run()

    def test_survae_overfit_batch(self):
        args = f'--model survae --root {InitializeExperiment.data_root} --dataset=mnist ' \
               f'--epochs={TestModels.overfit_epochs} --lr 1e-3 --gamma 0.995 --warmup 5000 --optimizer adamax ' \
               f'--num_steps 4 --num_scales 2 --dequant flow --pooling max --num_workers=0 ' \
               f'--batch_size=2 --training_class {InitializeExperiment.training_class} ' \
               f'--overfit --wandb_name_ext overfit'
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)
        TestModels._deactivate_evaluations(exp)

        exp.run()

    def test_iwae_overfit_batch(self):
        args = f'--model iwae --root {InitializeExperiment.data_root} --dataset=mnist ' \
               f'--epochs={TestModels.overfit_epochs} --lr 1e-4 ' \
               f'--num_k 16 --latent_dim 30 --num_workers=0 ' \
               f'--batch_size=2 --training_class {InitializeExperiment.training_class} ' \
               f'--overfit --wandb_name_ext overfit'
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)
        TestModels._deactivate_evaluations(exp)

        exp.run()

    ###############
    # Small Model
    ###############
    def test_survae_small_model(self):
        args = f'--model survae --dataset=mnist ' \
               f'--epochs={TestModels.small_model_epochs} --lr 1e-3 --gamma 0.995 --warmup 5000 --optimizer adamax ' \
               f'--num_steps 1 --num_scales 1 --dequant flow --pooling max --num_workers=0 ' \
               f'--batch_size=32 ' \
               f'--wandb_name_ext small_model ' + TestModels._get_eval_sample_training_class_text()
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()

    def test_nf_small_model(self):
        args = f'--model nf --root {InitializeExperiment.data_root} --flow=RealNVP --dataset=mnist ' \
               f'--epochs={TestModels.small_model_epochs} --lr=5e-5 ' \
               f'--num_blocks=1 --num_scales 1 --batch_size=32 ' \
               f'--num_workers=0 ' \
               f'--wandb_name_ext small_model ' + TestModels._get_eval_sample_training_class_text()
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()

    def test_iwae_small_model(self):
        # note that only the latent space is small here
        args = f'--model iwae --dataset=mnist ' \
               f'--epochs={TestModels.small_model_epochs} --lr 1e-4 ' \
               f'--num_k 16 --latent_dim 2 --num_workers=0 ' \
               f'--batch_size=32 ' \
               f'--wandb_name_ext small_model ' + TestModels._get_eval_sample_training_class_text()
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()

    ###############
    # Big Model
    ###############
    def test_survae_big_model(self):
        args = f'--model survae --dataset=mnist ' \
               f'--epochs={TestModels.big_model_epochs} --lr 1e-3 --gamma 0.995 --warmup 5000 --optimizer adamax ' \
               f'--num_steps 36 --num_scales 2 --dequant flow --pooling max --num_workers=0 ' \
               f'--batch_size=32 ' \
               f'--wandb_name_ext big_model ' + TestModels._get_eval_sample_training_class_text()
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()

    def test_nf_big_model(self):
        args = f'--model nf --flow=RealNVP --dataset=mnist ' \
               f'--epochs={TestModels.big_model_epochs} --lr=5e-5 ' \
                f'--num_blocks=2 --num_scales=2 --batch_size=32 ' \
                f'--num_workers=0 ' \
               f'--wandb_name_ext big_model ' + TestModels._get_eval_sample_training_class_text()

        # f'--num_blocks=6 --num_scales=5 --batch_size=32 ' \
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()

    def test_iwae_big_model(self):
        # note that only the latent space is big here
        args = f'--model iwae --dataset=mnist ' \
               f'--epochs={TestModels.big_model_epochs} --lr 1e-4 ' \
               f'--num_k 16 --latent_dim 256 --num_workers=0 ' \
               f'--batch_size=32 ' \
               f'--wandb_name_ext big_model ' + TestModels._get_eval_sample_training_class_text()
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()


    ###############
    # Hierarchical
    ###############
    def test_survae_hierarchical(self):
        args = f'--model survae --dataset=mnist ' \
               f'--epochs={TestModels.hierarchical_survae_epochs} --lr 1e-3 --gamma 0.995 --warmup 5000 --optimizer adamax ' \
               f'--num_steps 5 --num_scales 4 --dequant flow --pooling max --num_workers=0 ' \
               f'--batch_size=32 ' \
               f'--wandb_name_ext hierarchical ' + TestModels._get_eval_sample_training_class_text()
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=False)

        exp.run()

    @staticmethod
    def _get_eval_sample_training_class_text():
        return f'--eval_every {TestModels.eval_every} --root {InitializeExperiment.data_root} ' \
               f'--training_class {InitializeExperiment.training_class}'

    @staticmethod
    def _deactivate_evaluations(exp):
        exp.args.check_every = None
        exp.args.eval_every = None
        exp.args.sample_every = None
