import torch
import numpy as np
from matplotlib import pyplot as plt
from plotting.transformation import plot_data_space
from misc.naming import get_model_color
from flow_ssl.invertible.downsample import i_logits

from res_flow.res_flow_back_prob import res_flow_back_prob
from res_flow.master_utils import *

# Logging
from tqdm import tqdm

#from survae.model.distributions import DataParallelDistribution
from survae.model.data.path import get_survae_path

# Experiment
from master.training.base_training import BaseExperiment

# Master
from master.utils.utils import *
from master.training.utils import *
from master.utils.computational import evaluate_loader, evaluate_data_return_llh
from master.training.sanity_check import *
from misc.constants import *
from inflation.evaluate_model import compute_log_deflation


class FlowExperiment_Master(BaseExperiment):
    """This is the adjusted version of FlowExperiment from the SurVAE Paper"""

    def __init__(self, args, epochs, data_id, model_id,
                 train_loader, eval_loader, other_classes_loaders,
                 model, loss_fn, optimizer, scheduler_iter, scheduler_epoch,
                 logging, inflater, manifold_evaluation_data, grid_evaluation_data):

        # Init parent
        super(FlowExperiment_Master, self).__init__(args=args, epochs=epochs, model=model, optimizer=optimizer,
                                                    scheduler_iter=scheduler_iter, scheduler_epoch=scheduler_epoch,
                                                    eval_every=args.eval_every, logging=logging,
                                                    log_path=args.log_path)

        # Store args
        self.save_args(args)
        self.args = args
        self.loss_fn = loss_fn

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.other_classes_loaders = other_classes_loaders
        self.manifold_evaluation_data = manifold_evaluation_data
        self.grid_evaluation_data = grid_evaluation_data

        self.inflater = inflater
        self.clip_gradient = args.clip_gradient

        # losses for evaluation
        self.losses_per_sample = None

        if self.args.training_class == -1:
            self.args.training_class = self.args.dataset

        self._sanity_check()

    def pre_computation(self, batch):
        logdet = None
        if get_bool_from_args('pre_inflation_ilogits'):
            batch, logdet = i_logits(batch)
        return batch, logdet

    def train_fn(self, epoch):
        losses_per_sample = []  # this can later be used to make a quick evaluation
        data_points_count, loss_sum = 0, 0.0
        logdets = []
        start_time = time.time()

        if self.inflater is not None:
            self.inflater.update_noise_generator(epoch)

        self.model.train()
        for i, batch in tqdm(enumerate(self.train_loader)):
            batch = batch.to(DEVICE)
            # mainly nothing or e.g. iLogits
            batch, logdet = self.pre_computation(batch)

            if self.inflater is not None:
                batch = self.inflater(batch)

            self.optimizer.zero_grad()

            loss = self.loss_fn(self.model, batch, logdet=logdet)
            if loss is None:
                continue  # see discard_near_0_z

            loss_mean = loss.mean()
            loss_mean.backward()

            if self.args.model == 'res_flow':
                res_flow_back_prob(self)
            else:
                if self.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), max_norm=0.001)
                self.optimizer.step()

            # keeping track of overall losses
            with torch.no_grad():
                loss_sum += loss_mean.cpu().item() * len(batch)
                data_points_count += len(batch)
                logdets.append(self.model.logdet().detach())

            if self.scheduler_iter:
                self.scheduler_iter.step()

            if self.args.debug and data_points_count > 1000: break

        # train losses
        bpd = loss_sum / data_points_count

        # log 1/N sum exp( logdet_i ) = log ( sum exp(logdet_i - N ) )
        logdets = torch.cat(logdets)
        avg_logdet = torch.logsumexp(logdets - np.log(data_points_count), dim=0)

        # Output and log
        print(f'Train. Epoch: {epoch + 1}/{self.args.epochs} | '
              f'Datapoints: {data_points_count}/{len(self.train_loader.dataset)} | '
              f'llh: {-bpd*np.log(2) * np.prod(batch.shape[1:]):.3f} / bpd: {bpd:.3f} ' +
              f'Time {time.time() - start_time:.0f}s')

        self.losses_per_sample = losses_per_sample
        return {'loss': bpd, 'avg_logdet': avg_logdet}, bpd

    def evaluation(self, log):
        loss_msg = ''

        main_test_cls_loss, _, avg_logdet = evaluate_loader(self.model, self.loss_fn, self.eval_loader, self.args,
                                                            self.inflater, pre_computation=self.pre_computation)

        loss_msg += f'\t inflated main loss: {main_test_cls_loss:.3f} | '
        log['test_loss'] = main_test_cls_loss
        log['avg_logdet'] = avg_logdet

        if self.args.dataset == 'mnist':
            mani_loss, _, _ = evaluate_loader(self.model, self.loss_fn, self.eval_loader, self.args, None,
                                                pre_computation=self.pre_computation)
            loss_msg += f'\t manifold loss: {mani_loss:.2f} | '
            log['manifold_loss'] = mani_loss

        print(loss_msg)
        return main_test_cls_loss

    def eval_fn(self, epoch, log):
        print(f'Eval. Epoch: {epoch + 1}/{self.args.epochs} ', end='')

        if self.args.model == 'res_flow':
            set_resnet_eval(self.model)

        self.model.eval()
        with torch.no_grad():
            args_dir = dir(self.args)
            if 'use_new_loss' in args_dir and self.args.use_new_loss:
                self.model.compute_avg_log_volume(self.train_loader)
            elif 'new_loss_2' in args_dir and self.args.new_loss_2:
                self.model.compute_avg_log_volume_2()
            elif self.args.model == 'conv':
                self.model.compute_normalization(self.train_loader)

            result = self.evaluation(log)

            if self.args.model == 'res_flow':
                unset_resnet_eval(self.model)

        return result

    def sample_fn(self, epoch):
        # sampling
        if self.args.num_samples <= 0:
            return

        self.model.eval()
        with torch.no_grad():
            # write to file
            #sample_path = os.path.join(self.args.sample_root, f'sample_ep{epoch}.png')
            samples = self.model.sample(self.args.num_samples, manipulate_samples=True)
            if samples is None:
                warn('no samples created')
                return
            if self.args.dataset == 'mnist':
                samples[samples < 0] = 0.
                samples[samples > 1] = 1.

            samples = samples.cpu().float()
            #vutils.save_image(samples, fp=sample_path, nrow=AMOUNT_OF_COLS_FOR_LOG_IMAGES)

            # write to wandb
            grid = make_grid(samples, nrow=AMOUNT_OF_COLS_FOR_LOG_IMAGES)
            wand_img = wandb.Image(grid, caption=str(epoch))
            # wandb.log({'samples/samples by epoch': wand_img}, step=epoch, commit=False)
            # wandb.log({f'samples/{epoch}': wand_img}, commit=False)
            self.logging.log({f'samples': wand_img}, commit=False, step=epoch)

    def eval_manifold(self, epoch):
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)

        if self.args.model == 'res_flow':
            set_resnet_eval(self.model)

        self.model.eval()
        with torch.no_grad():
            true_log_pds = self.manifold_evaluation_data.densities[INFLATED_MANIFOLD_LOG_PD]
            points_on_manifold = self.manifold_evaluation_data.data[INFLATED_MANIFOLD_DATA]
            embedded_manifold = self.manifold_evaluation_data.data[EMBEDDED_DATA].cuda()

            embedded_manifold, log_det = self.pre_computation(embedded_manifold)

            bpd = self.loss_fn(self.model, embedded_manifold, mean=False, logdet=log_det)

        # remove bpd norming
        manifold_llh = -bpd * np.log(2) * self.args.dimensions
        # norm them
        std_manifold_llh = manifold_llh - manifold_llh.max()  # + true_log_pds.mean()
        # make sure the area under this curve is 1 (missing borders, but egal)
        std_manifold_llh = std_manifold_llh.exp()
        area_size = points_on_manifold[1] - points_on_manifold[0]
        normalizer = area_size * std_manifold_llh.sum()
        normed_to_one_lh = 1/normalizer * std_manifold_llh

        # plot likelihoods
        ax.plot(points_on_manifold, normed_to_one_lh.cpu(), label=self.args.flow,
                color=get_model_color(self.args.flow))

        # plot true data
        ax.plot(points_on_manifold, np.exp(true_log_pds), '--', label='true', color='k')

        ax.set_yticks([])
        ax.set_ylim(bottom=0, top=0.5)
        ax.set_ylabel('$\mathregular{q_M(m)}$', fontsize=20)
        ax.set_xlabel('m', fontsize=20)

        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=11)

        fig.savefig(self.log_path + f'/manifold/{epoch}.pdf', format='pdf', bbox_inches='tight')
        self.logging.log({f'manifold': wandb.Image(fig)}, commit=False, step=epoch)
        plt.close()

        if self.args.model == 'res_flow':
            unset_resnet_eval(self.model)

    def eval_grid(self, epoch):
        if self.args.model == 'res_flow':
            set_resnet_eval(self.model)

        assert False, 'No precomputation implemented'

        self.model.eval()
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)
        #ax.axis('equal')
        # data
        none_embedded_data = self.grid_evaluation_data.data[NONE_EMBEDDED_DATA]
        data_X, data_Y = none_embedded_data[:, 0], none_embedded_data[:, 1]
        #true_pd = self.manifold_evaluation_data.densities[FULL_INFLATED_LOG_PD]

        # eval
        embedded_data = self.grid_evaluation_data.data[EMBEDDED_DATA].cuda()

        with torch.no_grad():
            result = evaluate_data_return_llh(self.model, self.loss_fn, embedded_data, self.args)

        # plot
        # X, Y, pd, log, ax, var, true_pd, dataset, alpha, v_min=None, v_max=None, scatter_size=2,
        #                     no_circle=False, green_stuff=False
        result = (result / self.args.dimensions) * 2 # there are two dimensions left
        plot_data_space(data_X, data_Y, result, log=False, ax=ax, dataset=self.grid_evaluation_data,
        #plot_data_space(data_X, data_Y, result, log=self.args.dimensions > 2, ax=ax, dataset=self.grid_evaluation_data,
                        var=self.args.var, true_pd=None, alpha=0.01, scatter_size=0.8)

        # bounds
        bounds = self.grid_evaluation_data.get_inflated_manifold_bounds(3)
        ax.set_xlim(bounds)
        y_bound = max(0.5, 5 * self.args.normal_scale)
        ax.set_ylim(-y_bound, y_bound)

        self.logging.log({f'full_grid': wandb.Image(fig)}, commit=False, step=epoch)
        fig.savefig(self.log_path + f'/dist_change/{epoch}.png')
        plt.close()

        if self.args.model == 'res_flow':
            unset_resnet_eval(self.model)

    def finish(self):
        self.logging.finish()
        print_line()
        print('Finished Training')
        print_line()

    # this performs some sanity checks
    def _sanity_check(self):
        log_some_data(self.train_loader, self.eval_loader, self.other_classes_loaders)
