"""This generates images (e.g. all black, noise, ...) and computes the likelihood for them.
"""
from warnings import warn
from enum import Enum

# Helper Functions
from plotting.samples import *
from master.utils.utils import print_line


class ArtificialImages:

    def __init__(self, information_container):
        self.storage = information_container

    def run(self):
        print_line()
        print('ArtificialImages...')

        sample_storage = dict()
        sample_storage['bad'] = self._bad_images()
        sample_storage['good'] = self._good_images()
        return sample_storage

    def _bad_images(self):
        """I tried to create a couple of bad images"""
        data_shape = self.storage.data_shape

        info = [(torch.zeros(data_shape), 'black'),
                (torch.ones(data_shape), 'white'),
                (torch.ones(data_shape) * 0.5, '0.5 * white'),
                (torch.rand(data_shape) * 0.1, '0.1 * Uni Noise')]

        if EasyAccess().args.model == 'survae':
            info.append((torch.rand(data_shape), 'Uni Noise'))

        return self._img_to_lln_and_name(info)

    def _good_images(self):
        """I tried to create a couple of good images"""
        data_shape = self.storage.data_shape

        info = [(torch.zeros(data_shape), 'black')]

        for i in range(3):
            exp = -i - 3
            add = 10 ** exp
            info.append((torch.zeros(data_shape) + add, f'black+1e{exp}'))

        return self._img_to_lln_and_name(info)

    def _img_to_lln_and_name(self, info):
        """info is an array which elements of type (image, ...)"""
        result = []
        for img, name in info:
            img = img.unsqueeze(0)
            # produce likelihood
            lln = self.storage.loss_container.lln_detached(img)
            result.append((name, lln))

        return result

