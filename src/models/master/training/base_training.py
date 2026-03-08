import os
import pickle
import torch
import sys

from master.config import *

class BaseExperiment:
    """This module is adjusted from the BaseExperiment class of the SurVAE paper"""

    def __init__(self, args, epochs, model, optimizer, scheduler_iter, scheduler_epoch,
                 log_path, logging, eval_every):

        self.args = args
        self.epochs = epochs
        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch
        self.logging = logging

        # Paths
        self.log_path = log_path

        # Intervals
        self.eval_every = eval_every

        # Initialize
        self.current_epoch = 0
        self.prev_model_all_weights = None
        self.prev_model_common_weights = None
        self.prev_model_dependent_weights = None
        self.model_at_epoch_X = None
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []

    def train_fn(self, epoch):
        raise NotImplementedError()

    def eval_fn(self, epoch, eval_dict):
        raise NotImplementedError()

    def sample_fn(self, epoch):
        raise NotImplementedError()

    def eval_manifold(self, epoch):
        raise NotImplementedError()

    def finish(self):
        """This is called when all the training is done"""
        pass

    def log_fn(self, epoch, train_dict, eval_dict):
        for metric_name, metric_value in train_dict.items():
            self.logging.log({f'train/{metric_name}': metric_value}, step=epoch)

        if eval_dict:
            for metric_name, metric_value in eval_dict.items():
                self.logging.log({f'eval/{metric_name}': metric_value}, step=epoch)

    def log_train_metrics(self, train_dict):
        for metric_name, metric_value in train_dict.items():
            if metric_name not in self.train_metrics:
                self.train_metrics[metric_name] = []

            self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict, best_loss):
        for metric_name, metric_value in eval_dict.items():
            if metric_name not in self.eval_metrics:
                self.eval_metrics[metric_name] = []

            self.eval_metrics[metric_name].append(metric_value)

    def save_args(self, args):
        # Save args
        with open(os.path.join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

    def save_metrics(self):
        # Save metrics
        with open(os.path.join(self.log_path,'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(os.path.join(self.log_path,'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)

    def checkpoint_save(self, loss):
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}

        if 'ema' in dir(self.model):
            checkpoint['ema'] = self.model.ema

        if self.args.dont_check is False and not self.args.debug:
            print(f'Test improvement!  {self.args.check_path}')
            torch.save(checkpoint, os.path.join(self.args.check_path, 'checkpoint.pt'))

    def _get_model_params(self, weight_type):
        """weight_type: all_but_model_dependent
         only_model_dependent
         all"""
        weight_list = list()
        for name, weights in self.model.named_parameters():
            if weight_type == 'all_but_model_dependent':
                if 'store_for_resacles' in name or 'scale_layer' in name or 'rescale' in name \
                        or 'trainable_part' in name:  # very specifiy name for scale layer, ugly but fits for now
                    continue
                weight_list.append(weights.flatten().cpu())

            elif weight_type == 'model_dependent':
                if 'store_for_resacles' in name or 'scale_layer' in name or 'rescale' in name \
                        or 'trainable_part' in name:  # very specifiy name for scale layer, ugly but fits for now
                    weight_list.append(weights.flatten())

            elif weight_type == 'all':
                weight_list.append(weights.flatten().cpu())

        weight_list = torch.cat(weight_list)
        #print(f'For Test: len of weight list is {len(weight_list)}')
        return weight_list

    @staticmethod
    def _relative_distance(reference, current):
        reference = reference + torch.finfo(torch.float32).eps
        change = torch.abs((current - reference) / reference)
        return change.mean()

    @staticmethod
    def _relative_log_distance(reference, current):
        #reference += torch.finfo(torch.float32).eps
        not_close = torch.logical_not(torch.isclose(reference, current))
        current = current[not_close]
        reference = reference[not_close]
        change = torch.abs(torch.log(torch.abs(current / reference)))
        return change.mean()

    @staticmethod
    def _exp_abs_log_distance(reference, current):
        #reference += torch.finfo(torch.float32).eps
        not_close = torch.logical_not(torch.isclose(reference, current))
        current = current[not_close]
        reference = reference[not_close]
        change = torch.exp(torch.abs(torch.log(torch.abs(current / reference))))
        return change.mean()

    @staticmethod
    def _exp_abs_log_distance_2(reference, current):
        #reference += torch.finfo(torch.float32).eps
        #not_close = torch.logical_not(torch.isclose(reference, current))
        not_zero = torch.logical_not(torch.logical_or(torch.isclose(reference, torch.tensor(0.)),
                                                      torch.isclose(current, torch.tensor(0.))))
        current = current[not_zero]
        reference = reference[not_zero]
        change = torch.exp(torch.abs(torch.log(torch.abs(current / reference))))

        return change.mean()

    @staticmethod
    def _normed_weight_change_euc(reference, current):
        upper_euc_norm = torch.sqrt( torch.mean((current - reference)**2) )
        lower_euc_norm = torch.sqrt( torch.mean(reference**2) )

        change = upper_euc_norm / lower_euc_norm
        return change

    @staticmethod
    def _euclidean_percentual_change(reference, current):
        reference = reference + torch.finfo(torch.float32).eps

        change = ((current - reference) / reference) ** 2
        change = change.mean().sqrt()

        return change

    @staticmethod
    def _paper_weight_change(reference, current):
        upper = (current - reference).abs().mean()
        lower = reference.abs().mean()

        return upper / lower

    def eval_net_distance(self, eval_dict, epoch):
        with torch.no_grad():
            if self.prev_model_all_weights is not None:
                current_weights = self._get_model_params(weight_type='all')
                eval_dict['change_all'] = BaseExperiment._paper_weight_change(self.prev_model_all_weights,
                                                                            current_weights)

                """eval_dict['log_change_all'] = BaseExperiment._relative_log_distance(self.prev_model_all_weights,
                                                                                            current_weights)

                eval_dict['exp_abs_log_change_all'] = BaseExperiment._exp_abs_log_distance(self.prev_model_all_weights,
                                                                                            current_weights)

                eval_dict['exp_abs_log_change_all_2'] = BaseExperiment._exp_abs_log_distance_2(self.prev_model_all_weights,
                                                                                           current_weights)
                """
                eval_dict['euclidean_percentual_change'] = BaseExperiment._euclidean_percentual_change(self.prev_model_all_weights,
                                                                                                       current_weights)

                eval_dict['normed_euclidean_change'] = BaseExperiment._normed_weight_change_euc(
                                                                self.prev_model_all_weights,
                                                                current_weights)
                if self.model_at_epoch_X is not None:
                    eval_dict['rel_dist_to_model_100'] = BaseExperiment._paper_weight_change(self.model_at_epoch_X, current_weights)

                model_indep_weights = self._get_model_params(weight_type='all_but_model_dependent')
                eval_dict['change_common'] = BaseExperiment._paper_weight_change(self.prev_model_common_weights,
                                                                               model_indep_weights)

                model_dep_weights = self._get_model_params(weight_type='model_dependent')
                eval_dict['change_model_dependent'] = BaseExperiment._paper_weight_change(self.prev_model_dependent_weights,
                                                                                        model_dep_weights)

            self.prev_model_all_weights = self._get_model_params(weight_type='all')
            self.prev_model_common_weights = self._get_model_params(weight_type='all_but_model_dependent')
            self.prev_model_dependent_weights = self._get_model_params(weight_type='model_dependent')
            if epoch == 100:
                self.model_at_epoch_X = self._get_model_params(weight_type='all')

    def checkpoint_load(self, check_path):
        checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pt'))
        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

    def _print_infos(self):
        if 'mighty_prior' in dir(self.model):
            self.model.mighty_prior.print_weights()

    def run(self):
        self._train_iteration(self.epochs)
        self.finish()

    def _train_iteration(self, epochs):
        early_stopping = self.args.early_stopping
        best_train_loss = best_loss = float('inf')
        best_epoch = -1
        for epoch in range(self.current_epoch, epochs):
            eval_dict = dict()
            # Eval (first - I want to see the start performance)
            if (self.eval_every is not None and epoch % self.eval_every == 0) \
                    or self.args.debug:
                self._print_infos()
                main_test_cls_loss = self.eval_fn(epoch, eval_dict)
                # for the new case, we save checkpoints differently
                is_test_loss_better = main_test_cls_loss < best_loss
                if is_test_loss_better:
                    best_loss = main_test_cls_loss
                    self.checkpoint_save(loss=best_loss)
                eval_dict['best_test_loss'] = best_loss

                self.log_eval_metrics(eval_dict, best_loss)
                self.eval_epochs.append(epoch)

            # eval manifold
            if self.args.manifold_evaluation is not None and epoch % self.args.manifold_evaluation == 0:
                self.eval_manifold(epoch)
            if self.args.pdf_evaluation is not None and epoch % self.args.pdf_evaluation == 0:
                self.eval_grid(epoch)
            if self.args.distance_evaluation is not None and epoch % self.args.distance_evaluation == 0:
                self.eval_net_distance(eval_dict, epoch)

            # sampling
            if (self.args.sample_every is not None and epoch % self.args.sample_every == 0) or self.args.debug:
                self.sample_fn(epoch)

            # Train
            train_dict, train_loss = self.train_fn(epoch)

            if self.scheduler_epoch: self.scheduler_epoch.step()
            sys.stdout.flush()  # to avoid displaying problems

            # if the train loss hasn't improved for 30 epochs, terminate (if early_stopping is activated)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
            train_dict['best_train_loss'] = best_train_loss

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, eval_dict)

            self.current_epoch += 1

            if isinstance(early_stopping, int) and epoch - best_epoch >= early_stopping:
                print(f'Loss has not improved for {early_stopping} epochs. Terminate training.')
                break
