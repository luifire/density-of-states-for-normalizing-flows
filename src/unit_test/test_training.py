import unittest
import torch
from mocks.initialize_mocked_experiment import InitializeExperiment
from mocks.utils import clean_experiment
from easy_access import EasyAccess


class TestTraining(unittest.TestCase):
    """This mocks a model and tests the training environment"""

    def setUp(self):
        EasyAccess().reset()
        InitializeExperiment.initialize()

    def test_amount_of_calling(self):
        epochs = 5  # will be increased so we make sure the last evaluation is done as well
        eval_every = 1
        #check_every = 1

        args = f'--model nf --root {InitializeExperiment.data_root} --flow=RealNVP --dataset=mnist ' \
               f'--epochs={epochs} --eval_every {eval_every} --lr=5e-5 ' \
               f'--num_blocks=3 --batch_size=32 ' \
               f'--num_workers=0 --training_class 0'
        exp = InitializeExperiment().initialize_training(args)
        exp.run()

        epochs += 1
        eval_count = epochs // eval_every
        times_loss_gets_better = 1  # by custom loss function

        # epoch count
        self.assertEqual(exp.args.epochs, epochs)
        # model cals
        self.assertEqual(exp.model.train.call_count, epochs)
        self.assertEqual(exp.model.sample.call_count, eval_count)
        # our loss function is none increasing, so this happens
        self.assertEqual(exp.model.state_dict.call_count, times_loss_gets_better)

        # happens twice for each evaluation round
        # doesn't happen for new use case
        #self.assertEqual(exp.model.eval.call_count, 2*eval_count)

        # also called by evaluation, thus we use greater, this is quite a rough approximation
        # this function is also called for initialization
        self.assertGreaterEqual(exp.loss_fn.call_count, len(exp.train_loader) * epochs)

        # Optimizer
        should_optimizer = {'zero_grad': len(exp.train_loader) * epochs,
                            'step': len(exp.train_loader) * epochs,
                            'state_dict': times_loss_gets_better}
        for key, value in exp.optimizer.get_call_count().items():
            self.assertEqual(value, should_optimizer[key])

        clean_experiment(exp)

    ###############
    # Tests Loss
    # These tests always run the same batch through an freshly initialized net
    # the loss of this computation should always be the same.
    # the loss was recorded once
    ###############
    def test_nf_loss(self):
        args = f'--model nf --root {InitializeExperiment.data_root} --flow=RealNVP --dataset=mnist ' \
               f'--epochs=0 --lr=5e-5 --batch_size 1 ' \
               f'--num_blocks=3 --num_scales 2 ' \
               f'--num_workers=0 --training_class 0'
        exp = InitializeExperiment.initialize_training(args, mock_model=False)

        # old: 263.1812744140625
        # less old: 269.5872802734375
        self.assert_first_batch_yields(exp, 266.0984802246094)

        clean_experiment(exp)

    def test_survae_loss(self):
        args = f'--model survae --root {InitializeExperiment.data_root} --dataset=mnist ' \
               f'--epochs=0 --lr=5e-5 --batch_size 1 ' \
               f'--num_steps 1 --num_scales 1 --dequant flow --pooling max ' \
               f'--num_workers=0 --training_class 0'
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=True)

        # old: 11.164375305175781
        # less old: 11.156210899353027
        self.assert_first_batch_yields(exp, 11.163322448730469)
        clean_experiment(exp)

    def test_iwae_loss(self):
        args = f'--model iwae --root {InitializeExperiment.data_root} --dataset=mnist ' \
               f'--epochs=0 --lr=5e-5 --batch_size 1 ' \
               f'--num_k 16 --latent_dim 30 --num_workers 0 ' \
               f'--num_workers=0 --training_class 0'
        exp = InitializeExperiment.initialize_training(args, mock_model=False, mock_wandb=True)

        # old: 0.11702723801136017
        # without eval loader: 0.15036648511886597
        self.assert_first_batch_yields(exp, 0.17390665411949158)

        clean_experiment(exp)

    def assert_first_batch_yields(self, exp, expected_result):
        """Loads first batch, passes it through the net, compares the loss of that batch"""
        x, t = next(iter(exp.train_loader))
        x = x.cuda()
        with torch.no_grad():
            exp.model.eval()
            loss = exp.loss_fn(exp.model, x).mean().cpu().item()
            #print(loss)
            # this is never fully equal (also not with some epsilon) see:
            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            # thus we need to accept this threshold
            self.assertLessEqual(abs(loss-expected_result), 0.001)
