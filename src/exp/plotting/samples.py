########################
# Sampling and printing of the samples
########################

import numpy as np
import torch
from matplotlib import pyplot as plt

from master.config import *
from easy_access import EasyAccess
from shared_functions import model_short_to_name

from precomputation.pre_sampling import SampleType
from misc.naming import *
from misc.constants import *

from .utils import show_tensor_image


def plot_random_samples(samples, model_name, cols, amount=-1):
    samples = samples[model_name][SampleType.RANDOM]
    if amount > 0:
        samples = samples[:amount]
    _, axs = plt.subplots(len(samples) // cols, cols, figsize=(20, 4))
    for i, ax in enumerate(axs.reshape(-1)):
        show_tensor_image(samples[i], ax)

    plt.suptitle(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE, y=0.8)
    #plt.subplots_adjust(top=0.8, right=1.5)


def plot_unit_circle_image(all_samples, model_name, sample_type, cols):
    samples_with_likelihood = all_samples[model_name][sample_type]

    samples = samples_with_likelihood[0]
    llns = samples_with_likelihood[1]
    factors = samples_with_likelihood[2]

    _, axs = plt.subplots(len(factors) // cols, cols, figsize=(20, 4))
    axs = axs.reshape(-1)
    for sample, lln, factor, ax in zip(samples, llns, factors, axs):
        show_tensor_image(sample, ax)
        ax.set_title(f'({factor:.2f}, {lln:.3f})')

    #plt.subplots_adjust(right=1.1)
    print("(a, log p(x_a) in bpd)")
    plt.suptitle(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE)


def plot_random_samples_with_factor(all_samples, model_name, sample_type, row_count):
    samples_with_likelihood = all_samples[model_name][sample_type]
    lines_of_images = samples_with_likelihood[0]
    factors = samples_with_likelihood[1]

    rows = min(row_count, len(lines_of_images))
    cols = len(factors)
    fig, axs = plt.subplots(rows, cols)#, figsize=(rows*2, 7))
    for y in range(rows):
        for x in range(cols):
            show_tensor_image(lines_of_images[y][x], axs[y, x])
            if y == 0:
                axs[0, x].set_title(f'{factors[x]:.2f}')

    fig.subplots_adjust(right=1.3)
    plt.suptitle(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE)
    #print('-> Value of all z\'s')


def show_images_with_lln(all_samples, model_name, sample_type, cols, print_line=None):
    """This function prints images together with their losses.
    Cols is the amount of columns that should be used
    print_line is used to print a specific line (needed to demonstrate 0's)"""
    samples_with_likelihood = all_samples[model_name][sample_type]
    samples = samples_with_likelihood[0]
    lln = samples_with_likelihood[1]
    sample_count = len(samples)
    fig, axs = plt.subplots(sample_count // cols, cols, figsize=(20, 4))

    # compute losses
    samples = samples.float().cpu()

    for i, ax in enumerate(axs.reshape(-1)):
        img = samples[i]
        log_prob = lln[i]

        #  print image
        show_tensor_image(img, ax)
        ax.set_title(f'{log_prob:.2f}')

    print(f'Average log(p(x)) in bpd: {lln.mean():.4f}')
    plt.suptitle(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE, y=0.8)
    #fig.subplots_adjust(right=1.5)

    if print_line is not None:
        print('Line 15 of the first Image:')
        print(samples_with_likelihood[0][0, 0, print_line])
