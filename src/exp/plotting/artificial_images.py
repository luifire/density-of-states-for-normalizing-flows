########################
# This contains functions to find outliers
########################

from .utils import *


def plot_lln_of_artificial_images(artificial_images, good_or_bad, interesting_losses, best_imgs,
                                  worst_imgs=None, sharex=False):
    # when there are more additional images, we need to increase the plot size
    # yes this is ugly
    additional_length = max([len(x) for x, _ in artificial_images.items()])
    height_increase = additional_length + len(interesting_losses) - 7
    if height_increase < 0:
        height_increase = 0
    axs = print_class_averages(interesting_losses, sharex=sharex, height_increase=height_increase)

    for i, model in enumerate(artificial_images):
        # to obtain different colors
        reset_color_iterator()

        bottom_top_images = best_imgs[model].copy()
        if worst_imgs is not None:
            bottom_top_images.update(worst_imgs[model])

        sorted_losses = sorted(bottom_top_images)
        vline(sorted_losses[-1], 'best sample', ax=axs[i])
        if worst_imgs is not None:
            vline(sorted_losses[0], 'worst sample', ax=axs[i])

        for name, loss in artificial_images[model][good_or_bad]:
            color = 'k' if name == 'black' else None
            vline(loss, name, color=color, ax=axs[i])

        axs[i].legend(fontsize=LEGEND_FONT_SIZE)

    plt.xlabel('log p(x) in bpd', fontsize=AXIS_FONT_SIZE)
    plt.show()
