"""
This module was adapted from model
"""

import math
import warnings
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, CenterCrop
from torchvision.transforms import ToTensor

from survae.model.data.loaders.image import CIFAR10, ImageNet32, ImageNet64, SVHN
from master.data.mnist_master import MNIST_MASTER
from master.data.fashion_mnist_master import Fashion_MNIST
from master.data.sphere_data import VonMises
from master.data.line_data import LineData
from master.data.gauss_data import GaussData
from master.data.smoothstep_data import SmoothstepData
from master.data.rotated_shapes import RotatedData
from master.data.train_class_wrapper import TrainClassWrapper

from easy_access import EasyAccess
from master.config import *
from misc.constants import *


simple_data_choice = {'sphere', 'line', 'sigmoid', 'gauss', 'smooth_step'}
dataset_choices = {'cifar10', 'imagenet32', 'imagenet64', 'svhn', 'mnist', 'fashionmnist'}
dataset_choices.update(simple_data_choice)


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='cifar10', choices=dataset_choices)

    # Train params
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)


def get_data_id(args):
    return '{}_{}bit'.format(args.dataset, args.num_bits)


def get_data(args, main_class, dataset_name=None, shuffle_train=True, verbose=True, data_mode='sampled_manifold',
             ): # manifold_noise_scale=None
    """manifold_noise_scale is added during draining. but needed when this data is loaded otherwise"""
    assert args.dataset in dataset_choices

    if dataset_name == None:
        dataset_name = args.dataset

    data_path = args.data_path
    # Dataset
    data_shape = get_data_shape()
    pil_transforms, eval_transformation = get_augmentation(args, args.augmentation, args.dataset, data_shape)
    if dataset_name == 'cifar10':
        ...
        #dataset = CIFAR10(root=data_path, num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif dataset_name == 'imagenet32':
        ...
        #dataset = ImageNet32(root=data_path, num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif dataset_name == 'imagenet64':
        ...
        #dataset = ImageNet64(root=data_path, num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif dataset_name == 'svhn':
        ...
        #dataset = SVHN(root=data_path, num_bits=args.num_bits, pil_transforms=pil_transforms)
    elif dataset_name == 'mnist':
        dataset = MNIST_MASTER(root=data_path, num_bits=args.num_bits, download=True, pil_transforms=pil_transforms,
                               eval_transformation=eval_transformation)
        # hack to get it into normal shape
        dataset.train = dataset.train.data.unsqueeze(1) / 255.
        dataset.test = dataset.test.data.unsqueeze(1) / 255.
        dataset.dataset_creator = None  # nothing creates MNIST. This info is used later to not apply the rotation matrix
    elif dataset_name == 'fashionmnist':
        dataset = Fashion_MNIST(root=data_path, num_bits=args.num_bits, download=True, pil_transforms=pil_transforms,
                               eval_transformation=eval_transformation)
    else:
        rotation_args = dict(dimension=args.dimensions, size=args.train_size,
                             data_mode=data_mode) # , normal_noise_scale=manifold_noise_scale # noise will be randomly added during training
        if args.no_rotation:
            rotation_args['rotation_seed'] = None

        if dataset_name == 'sphere':
            dataset = TrainClassWrapper(dataset_name, sphere_radius=args.sphere_radius, mises_kappa=args.mises_kappa, **rotation_args)
        elif dataset_name == 'line':
            dataset = TrainClassWrapper(dataset_name, line_length=args.line_length, **rotation_args)
        elif dataset_name == 'smooth_step':
            dataset = TrainClassWrapper(dataset_name, start=args.data_start, end=args.data_end, offset=args.data_offset,
                                     Range=args.data_range, growth=args.data_growth, **rotation_args)
        elif dataset_name == 'gauss':
            dataset = TrainClassWrapper(dataset_name, **rotation_args)

    #if dataset_name not in simple_data_choice:
    #    dataset = _remove_classes(dataset, main_class)

    # if we should overfit, we cut down the training set here
    """if args.overfit:
        warnings.warn('Overfitting to first batch!')
        # we repeat the first batch "repeat"-times.
        # We assume that 32 would be the normal batch size
        repeat = len(dataset.train) // 32

        dataset.train.data = dataset.train.data[0:args.batch_size].repeat(repeat, 1, 1)
        dataset.train.targets = dataset.train.targets[0:args.batch_size].repeat(repeat)
        # don't shuffle on overfit
        shuffle_train = False"""

    # Data Loader
    train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=shuffle_train,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_loader = DataLoader(dataset.test, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)

    if verbose:
        print(f'Class: {main_class} | Train Data: {len(train_loader.dataset)} | Test Data {len(eval_loader.dataset)}')

    return train_loader, eval_loader, data_shape, dataset.dataset_creator


def create_evaluation_loaders(args, use_train_loaders=True, verbose=True):
    """Creates all loaders that is used for evaluation of other classes"""
    eval_loaders = dict()

    if args.dataset == 'fashionmnist':
        unshuffled_train_loader, eval_loader, _ = get_data(args, -1, dataset_name='mnist', shuffle_train=False, verbose=verbose)
        if use_train_loaders:
            eval_loaders['mnist'] = unshuffled_train_loader
        else:
            eval_loaders['mnist'] = eval_loader

    elif args.dataset == 'mnist':
        ...
        """# this is the old use case where we do within mnist ood detection
        for i in range(get_class_count(args.dataset)):
            if i == args.training_class:
                continue

            unshuffled_train_loader, eval_loader, _ = get_data(args, i, shuffle_train=False, verbose=verbose)
            if use_train_loaders:
                eval_loaders[i] = unshuffled_train_loader
            else:
                eval_loaders[i] = eval_loader"""
    elif args.dataset in simple_data_choice:
        # no test set
        ...

    return eval_loaders


def get_class_count(dataset_name):
    if dataset_name == 'mnist':
        return 10
    elif dataset_name == 'fashionmnist':
        return 10
    else:
        raise NotImplementedError('Class Count for other datasets')


def is_dataset_colored(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in ['cifar10', 'svhn']:
        return True
    #elif dataset_name in ['mnist', 'fashionmnist', 'sphere', 'line']:
    else:
        return False

    assert False, 'Unknown dataset'


def _create_dirty_data_set_for_attribute(complete_partition, new_partition, main_class):
    """complete_partition: complete train or eval set
    new_partition: new train or eval set
    we will append dirty data to the train and the eval set
    """
    unique_class_labels = complete_partition.targets.unique()

    # resolve B / (A + B) = 1% to B
    total_dirt_images = len(new_partition.data) * DIRT_IN_DATASET / (1 - DIRT_IN_DATASET)
    # -1 as the main class is in this as well
    dirt_images_per_class = total_dirt_images / (len(unique_class_labels) - 1)
    dirt_images_per_class = int(np.floor(dirt_images_per_class))

    for cls in unique_class_labels:
        # don't append main class
        if cls == main_class:
            continue

        # remove unwanted classes
        filter = complete_partition.targets == cls
        dirty_data = complete_partition.data[filter]
        dirty_targets = complete_partition.targets[filter]

        # adjust size
        dirty_data = dirty_data[:dirt_images_per_class]
        dirty_targets = dirty_targets[:dirt_images_per_class]

        # concat to training data
        new_partition.data = torch.cat((new_partition.data, dirty_data))
        new_partition.targets = torch.cat((new_partition.targets, dirty_targets))


def _create_dirty_data_set(complete_dataset, main_class):
    """99% main class 1% dirt
    1% dirt is equally distributed over all other classes"""

    new_dataset = _remove_classes(complete_dataset, main_class)

    _create_dirty_data_set_for_attribute(complete_dataset.train, new_dataset.train, main_class)
    # probably not used
    #_create_dirty_data_set_for_attribute(complete_dataset.test, new_dataset.test, main_class)

    return new_dataset


def _remove_unwanted_classes(dataset, class_filter):
    """Removes all samples of a class
    class_filter can be a filter function or an integer.
    we will only keep the samples filtered by class_filter """

    if type(class_filter) == int:
        class_number = class_filter
        if class_number == -1:
            # for -1 accept every class (for test purposes)
            class_filter = lambda data: data == data
            warnings.warn('Use all Classes!!!!')
        else:
            class_filter = lambda data: data == class_number

    train = dataset.train
    test = dataset.test

    train_filter = class_filter(train.targets)
    train.data = train.data[train_filter]
    train.targets = train.targets[train_filter]

    test_filter = class_filter(test.targets)
    test.data = test.data[test_filter]
    test.targets = test.targets[test_filter]

    dataset.train = train
    dataset.test = test


def _remove_classes(dataset, class_filter):
    """Removes all samples of a class
    class_filter can be a filter function or an integer.
    we will only keep the samples filtered by class_filter """

    if type(class_filter) == int:
        class_number = class_filter
        if class_number == -1:
            # for -1 accept every class (for test purposes)
            class_filter = lambda data: data == data
            warnings.warn('Use all Classes!!!!')
        else:
            class_filter = lambda data: data == class_number

    new_dataset = copy.deepcopy(dataset)

    train = new_dataset.train
    test = new_dataset.test

    train_filter = class_filter(train.targets)
    train.data = train.data[train_filter]
    train.targets = train.targets[train_filter]

    test_filter = class_filter(test.targets)
    test.data = test.data[test_filter]
    test.targets = test.targets[test_filter]

    new_dataset.train = train
    new_dataset.test = test

    return new_dataset


def get_augmentation(args, augmentation, dataset, data_shape):
    c, h, w = data_shape

    eval_transformation = []
    train_transforms = []

    if augmentation == 'horizontal_flip':
        train_transforms += [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'neta':
        train_transforms += [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04))]
    elif augmentation == 'eta':
        if dataset != 'mnist':
            train_transforms += [RandomHorizontalFlip()]
        train_transforms += [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04))]

    #elif args.model == 'nf' or args.model == 'iwae' or args.model == 'demun' :
    train_transforms += [CenterCrop(h), ToTensor()]
    eval_transformation += [CenterCrop(h), ToTensor()]

    return train_transforms, eval_transformation


def get_data_shape():
    dataset = EasyAccess().args.dataset
    if dataset == 'cifar10':
        return (3,32,32)
    elif dataset == 'imagenet32':
        return (3,32,32)
    elif dataset == 'imagenet64':
        return (3,64,64)
    elif dataset == 'svhn':
        return (3,32,32)
    elif dataset == 'mnist':
        if EasyAccess().args.model == 'survae':
            #print('avoid this datashape')
            return (1, 32, 32)
        return (1,28,28)
    elif dataset == 'fashionmnist':
        return (1, 28, 28)
    elif dataset in simple_data_choice:
        args = EasyAccess().args
        if 'list_dimension' in dir(args) and args.list_dimension:
            return (1, 1, args.dimensions)
        else:
            width = int(np.sqrt(args.dimensions))
            return (1, width, width)
        #return (1, 1, EasyAccess().args.dimensions)
