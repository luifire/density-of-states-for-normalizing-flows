########################
# This module is about plotting stuff
########################
import numpy as np
import torch

# Plot Stuff
import seaborn as sns
from IPython.display import clear_output

# Master Stuff
from misc.constants import *
from misc.naming import *
from easy_access import EasyAccess
from shared_functions import model_short_to_name
from .utils import compute_quantile


def show_outlier_loss_histogram(all_models, lim_left=None, lim_right=None, score_idx=None, quantile=0.01):
    """Makes the main histogram plot for the new use case"""
    height = len(all_models) * 5
    fig, axs = plt.subplots(nrows=len(all_models), sharey=True, squeeze=False, figsize=(PLOT_LENGTH, height))
    axs = axs.flatten()

    for i, key in enumerate(all_models):
        show_outlier_loss_histogram_for_model(all_models[key], key, ax=axs[i], lim_left=lim_left, lim_right=lim_right,
                                   score_idx=score_idx, quantile=quantile)

    if score_idx is None or score_idx == PD_IDX:
        plt.xlabel('log p(x) in bpd', fontsize=TITLE_FONT_SIZE)
    elif score_idx == T_SCORE:
        plt.xlabel('t-score( log p(x) in bpd )', fontsize=TITLE_FONT_SIZE)
    else:
        plt.xlabel('What is dis?')


def show_outlier_loss_histogram_for_model(all_losses, model_name, ax=None,
                                          lim_left=None, lim_right=None, score_idx=None, quantile=0.01):
    if ax is None:
        _, ax = plt.subplots(nrows=1, figsize=(PLOT_LENGTH, 5))

    if score_idx is None:
        score_idx = PD_IDX

    args = EasyAccess().args

    for cls, losses in all_losses.items():
        # not interested in entire plot
        if cls == MAIN_TRAIN_CLS:
            continue

        lln, color = losses[score_idx], get_class_color(cls)
        label = get_legend_label(cls, lln.mean())
        if cls == args.training_class or cls == MAIN_TRAIN_CLS:
            # main class: thicker, no histogram
            sns.distplot(lln, ax=ax, label=label, hist=False, kde_kws={'linewidth': 3}, color=color)
        else:
            sns.distplot(lln, ax=ax, label=label, hist=True, color=color,
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 0.1})

    quantile_value = compute_quantile(all_losses, q=quantile, score_idx=score_idx)
    vline(quantile_value, f'{int(quantile*100)}% quantile', ax=ax, color='k', set_small_label=False)

    if lim_left is not None:
        ax.set_xlim(left=lim_left)
    if lim_right is not None:
        ax.set_xlim(right=lim_right)

    ax.set_ylabel('Densitiy', fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel('')
    ax.set_title(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    clear_output()

    return ax


def show_model_loss_histograms(all_losses, model_name, lim_left=None, lim_right=None, ax=None, score_idx=None):
    """This prints the histogram of losses
    lim_left / lim_right are for plot.xlim
    score_idx: in case you want a different index than LLH_IDX"""

    if ax is None:
        _, ax = plt.subplots(nrows=1, figsize=(PLOT_LENGTH, 5))

    if score_idx is None:
        score_idx = PD_IDX

    args = EasyAccess().args
    for cls, losses in all_losses.items():
        lln = losses[score_idx]
        # Make training class thick
        kde_kws = dict()
        if cls == args.training_class or cls == MAIN_TRAIN_CLS:
            kde_kws['linewidth'] = 3

        color = get_class_color(cls)
        sns.distplot(lln, ax=ax, label=get_legend_label(cls, lln.mean()), hist=False, kde_kws=kde_kws, color=color)

    if lim_left is not None:
        ax.set_xlim(left=lim_left)
    if lim_right is not None:
        ax.set_xlim(right=lim_right)

    ax.set_ylabel('Densitiy', fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel('')
    ax.set_title(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    clear_output()

    return ax


def show_tensor_image(tensor, ax):
    """This prints a tensor to axis ax"""
    img = tensor.numpy()
    ax.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def _print_class_averages(all_losses, model_name, ax):
    reset_color_iterator()

    # shows the average losses of our classes
    for cls, losses in all_losses.items():  # f'Avg loss cls: {class_to_name(cls)}'
        lln = losses[PD_IDX]
        mean = lln.mean()
        vline(mean, label=get_legend_label(cls, mean), ax=ax, color=get_class_color(cls), set_small_label=False)
        ax.text(mean, 1.005, class_to_name(cls), color=get_class_color(cls), fontsize=6)

    ax.set_title(model_short_to_name(model_name), fontsize=AXIS_FONT_SIZE)
    ax.set_yticks([])
    ax.legend(fontsize=LEGEND_FONT_SIZE)


def print_class_averages(all_losses, sharex=True, height_increase=0, lim_left=None, lim_right=None):
    """Iterates over every model and prints them to their axis using __print_class_averages"""
    height = len(all_losses) * 5
    _, axs = plt.subplots(nrows=len(all_losses), sharex=sharex, squeeze=False,
                          figsize=(PLOT_LENGTH, height+height_increase), dpi=100)
    axs = axs.flatten()

    for i, key in enumerate(all_losses):
        _print_class_averages(all_losses[key], key, ax=axs[i])

    plt.xlabel('log p(x) in bpd', fontsize=AXIS_FONT_SIZE)

    if lim_left is not None:
        plt.xlim(left=lim_left)
    if lim_right is not None:
        plt.xlim(right=lim_right)

    return axs


color_idx = 0
def vline(loss, label, ax=None, color=None, set_small_label=True):
    # Print a vertical line and pick a new color for each line
    # this part just picks a new color
    global color_idx
    if color is None:
        color = plt.get_cmap('Dark2').colors[color_idx]
        color_idx += 1

    if ax is None:
        ax = plt

    ax.axvline(loss, linewidth=2, color=color, label=label)
    if set_small_label:
        ax.text(loss, 1.005, label, color=color, fontsize=6)


def reset_color_iterator():
    global color_idx
    color_idx  = 0


def initialize_loss_plot():
    global color_idx
    plt.figure(figsize=(20, 5), dpi=160)
    color_idx = 0
    #set_seeds(42)

    plt.yticks([])
    plt.xlabel('log p(x) in bpd', fontsize=AXIS_FONT_SIZE)


def print_examples_from_loaders(loaders, amount):
    """This prints the first 'amount' of images from each loader to the output"""
    fig, axs = plt.subplots(len(loaders), amount)

    for i, (cls, loader) in enumerate(loaders.items()):
        x = next(iter(loader))
        for j in range(amount):
            show_tensor_image(x[j], axs[i, j])
            if j == amount - 1:
                axs[i, j].set_title(class_to_name(cls), x=4, y=0)

    fig.subplots_adjust(top=1.2)


def show_outliers(outliers, cols, model_name, desc=False, verbose=True):
    """where outliers is a dictionary with outliers[loss] = (class, image)"""

    # sort by loss
    sorted_losses = sorted(outliers)
    if desc:
        sorted_losses = sorted_losses[::-1]

    rows = len(outliers) // cols + 1
    _, axs = plt.subplots(rows, cols)#, figsize=(rows*3, cols*1.3))
    axs = axs.reshape(-1)
    for i, loss in enumerate(sorted_losses):
        cls, img = outliers[loss]
        show_tensor_image(img.unsqueeze(0), axs[i])
        float_img = img.type(torch.cuda.FloatTensor)
        #unequal_neighbors = compute_unequal_neighbor(float_img.unsqueeze(0))
        if verbose:
            title = f'({loss:.2f}, {float_img.mean():.0f}, {class_to_name(cls)})'
        else:
            title = f'{loss:.2f}'
        axs[i].set_title(title, y=0.98, fontsize=10)

    # remove unused subplots
    for i in range(len(sorted_losses), len(axs)):
        plt.delaxes(axs[i])

    big_plot = rows > 4

    if big_plot:
        plt.subplots_adjust(right=1.7, top=1.9)
        title_pos = 2
    else:
        plt.subplots_adjust(right=1.7, top=1.3)
        title_pos = 1.4

    if verbose:
        print('Title = (log p(x) in bpd, pixel mean, class)')
    else:
        print('Title = log p(x) in bpd')
    plt.suptitle(model_short_to_name(model_name), fontsize=TITLE_FONT_SIZE, y=title_pos)


def _plot_histogram_and_likelihoods(histogram_losses, line_losses, lim_left=None, lim_right=None):
    fig = plt.figure(figsize=(20, 5))
    show_loss_histograms(histogram_losses, lim_left, lim_right, fig)

    for loss in line_losses:
        plt.axvline(loss, linewidth=0.9, c=get_class_color(line_losses[loss][0]))

    if lim_left is not None:
        plt.xlim(left=lim_left)
    if lim_right is not None:
        plt.xlim(right=lim_right)


def plot_histogram_and_likelihoods(interesting_losses, individual_losses, lim_left=None, lim_right=None):
    fig, axs = show_loss_histograms(interesting_losses, lim_left, lim_right)

    for i, model in enumerate(interesting_losses):
        for loss in individual_losses[model]:
            axs[i].axvline(loss, linewidth=0.9, c=get_class_color(individual_losses[model][loss][0]))

