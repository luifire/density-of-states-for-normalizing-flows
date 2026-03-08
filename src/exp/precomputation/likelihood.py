"""This computes the likelihood of images from the loaders
"""

from tqdm import tqdm

# Helper Functions
from precomputation.initialization import *
from misc.naming import *
from master.utils.utils import print_line
from survae.experiments.image.utils import set_seeds


class PrecomputeLikelihood:

    def __init__(self, information_container):
        self.storage = information_container

    def run(self):
        """Goes over all classes"""
        print_line()
        print('Computing log likelihoods')
        loss_of_all_classes = dict()
        for cls, loader in self.storage.all_loaders.items():
            # Debug Stuff: skip the non interesting classes
            if self.storage.debug and cls not in self.storage.debug_interesting_classes:
                continue

            print(f'Compute loss for {class_to_name(cls)}')
            latent_info = self._compute_sample_likelihood(loader)

            loss_of_all_classes[cls] = latent_info

        return self._compute_t_score(loss_of_all_classes)

    def _compute_t_score(self, loss_of_all_classes):
        # there is only the MAIN_TRAIN_CLS left
        # this is due the new use case
        all_losses = loss_of_all_classes[MAIN_TRAIN_CLS]

        # we filter the previously computed train loader
        # s.t. we will have the individual classes separated
        targets = all_losses[TARGET_IDX]

        # T-Score (T = (lln - mean) / std_dev)
        mean = all_losses[PD_IDX].mean().item()
        std_dev = all_losses[PD_IDX].var().item() ** 0.5
        # for the entire class
        loss_of_all_classes[MAIN_TRAIN_CLS][T_SCORE] = (loss_of_all_classes[MAIN_TRAIN_CLS][PD_IDX] - mean) / std_dev

        for cls in loss_of_all_classes:
            loss_of_all_classes[cls][T_SCORE] = (loss_of_all_classes[cls][PD_IDX] - mean) / std_dev

        return loss_of_all_classes

    def _compute_sample_likelihood(self, loader):
        """This evaluates all samples in the loader and stores the likelihood for each
        sample.
        If set_seed is True, we set the seed to config.seed

        info_per_result: Some loss functions can return more than one value (e.g. lln and z).
                         This number tells us how many there are.
        """
        torch.no_grad()
        config = self.storage.config
        if self.storage.reset_seed:
            set_seeds(config.seed)

        # create storage as dict
        all_keys = [IMG_IDX, PD_IDX, TARGET_IDX]
        if self.storage.STORES_FULL_INFORMATION:
            all_keys += [Z_IDX, VOLUME_IDX]

        all_losses = {key: [] for key in all_keys}

        for i, (x, target) in tqdm(enumerate(loader)):
            # stores the index of the given images (yes, this is ugly.
            # Alternatively I would store the images next to the information.
            # But this would be to space consuming)
            # Also need to use batch_size as the last batch will have a different size!
            all_losses[IMG_IDX].append(i * config.batch_size + torch.arange(len(x)))
            all_losses[TARGET_IDX].append(target)
            x = x.to(DEVICE)
            # we make it negative to get the true lln (because loss_fn creates - lln for gradient descent)
            loss_gpu = self.storage.loss_fn(x)
            # if we return one value, we don't have a tuple
            if self.storage.STORES_FULL_INFORMATION:
                all_losses[PD_IDX].append(loss_gpu[0])
                all_losses[Z_IDX].append(loss_gpu[1])
                all_losses[VOLUME_IDX].append(loss_gpu[2])
            else:
                all_losses[PD_IDX].append(loss_gpu)

            if self.storage.debug and i >= config.batch_count:
                break

        # concat everything (by information)
        result = dict()
        for key in all_losses: # range(self.storage.info_per_loss_result + 1):
            concated_info = torch.cat(all_losses[key])
            concated_info = concated_info.cpu().numpy()

            result[key] = concated_info

        return result
