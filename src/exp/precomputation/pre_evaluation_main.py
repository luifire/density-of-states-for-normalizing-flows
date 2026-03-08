"""This module does all the preprocessing before the evaluation happens.
This includes loading and preprocessing of images, feeding them through the deep gen model
and storing all relevant information.
something like:
image index,
image class (e.g. 0 xor 1 ... xor 9 for MNIST)
log likelihood
optional: z
optional: volume change / probability contribution
"""

import argparse
from warnings import warn

# Helper Functions
from precomputation.initialization import *
from misc.naming import *
from misc.common_functions import load_config
from master.utils.utils import print_line
from survae.experiments.image.utils import set_seeds

# Precomputation Stuff
from precomputation.likelihood import PrecomputeLikelihood
from precomputation.pre_sampling import PreSampling
from precomputation.artificial_images import ArtificialImages


class PreEvaluationMain:
    # if True stores (lln, z, volume), if False stores (lln)
    STORES_FULL_INFORMATION = True
    # name of the file where we store the information in (relative to checkpoint path)
    FILE_NAME = 'proprocessed_evaluation.pickle'
    # For Debug Purposes, we only compute likelihoods and samples for these classes
    DEBUG_INTERESTING_CLASSES = [MAIN_TRAIN_CLS, 1, 3, 4, 7, 8]

    def __init__(self, checkpoint, data_path, debug):
        self.checkpoint = checkpoint
        self.debug = debug
        print(f'Checkpoint: {self.checkpoint}')

        # config
        config = load_config(self.checkpoint, data_path)
        self.config = create_evaluation_args(config, debug)

        self._manipulate_config()

        # special classes
        self.MAIN_TEST_CLS = self.config.training_class
        self.debug_interesting_classes = [self.MAIN_TEST_CLS] + PreEvaluationMain.DEBUG_INTERESTING_CLASSES

        set_seeds(self.config.seed)
        # Data
        self.all_loaders = load_data(self.config)
        # Model
        self.model = load_model(self.checkpoint)

        if 'new_loss' in dir(self.config) and self.config.new_loss:
            print('computing average volume')
            self.config.avg_log_volume = self.model.compute_avg_log_volume(self.all_loaders[MAIN_TRAIN_CLS])
        if self.config.model == 'conv':
            self.model.compute_normalization(self.all_loaders[MAIN_TRAIN_CLS])

        # Loss Function
        self.loss_container = load_loss(self.all_loaders[MAIN_TRAIN_CLS])
        if PreEvaluationMain.STORES_FULL_INFORMATION:
            self.loss_fn = self.loss_container.x_to_full_info
            self.info_per_loss_result = 4
        else:
            self.loss_fn = self.loss_container.lln
            self.info_per_loss_result = 2

        # Shapes (needs to happen after model loading)
        self.data_shape, self.latent_shape, self.total_dim = get_shapes()

        # Other Settings
        self.reset_seed = True  # resets the seed for every loader (makes things easier in the evaluation)

    def _manipulate_config(self):
        """Due to some data structure changes, we need to manipulate the config"""
        print_line()
        if self.config.model == 'iwae' and 'kld_weight' not in dir(self.config):
            self.config.kld_weight = None  # this is actually irrelevant for evaluation

        if 'no_bias' not in dir(self.config):
            if 'no_bias' in self.checkpoint:
                self.config.no_bias = True
                print('Set no_bias = False')
            else:
                self.config.no_bias = False

        if 'new_loss' not in dir(self.config):
            if 'new_loss' in self.checkpoint:
                self.config.new_loss = True
                print('Set new_loss = True')
            else:
                self.config.new_loss = False
        print_line()

    def _pack_additional_info(self):
        return {'data_shape': self.data_shape,
                       'total_dim': self.total_dim,
                       'latent_shape': self.latent_shape,
                       'total_latent_dim': np.prod(self.latent_shape),
                       'avg_log_volume': self.config.avg_log_volume if 'avg_log_volume' in self.config else None}

    def run(self):
        bag = dict()
        warn('model to eval mode')
        warn('!!!!!!!!!!!!!!!')
        bag['samples'] = PreSampling(self).run()
        bag['artificial'] = ArtificialImages(self).run()
        bag['lln'] = PrecomputeLikelihood(self).run()
        bag['additional'] = self._pack_additional_info()

        self._store_processed_information(bag)

    def _store_processed_information(self, bag):
        file_name = path.join(self.checkpoint, PreEvaluationMain.FILE_NAME)
        print_line()
        print(f'Storing Information to: {file_name}')
        with open(file_name, 'wb') as handle:
            pickle.dump(bag, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Folder with all relevant checkpoint information. E.g.:'
                                                       '.../RealNVP_6_mnist_cls-0/2021-04-23_15-59-06')
    parser.add_argument('--data', type=str, help='Folder containing all data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    PreEvaluationMain(args.checkpoint, args.data, args.debug).run()
