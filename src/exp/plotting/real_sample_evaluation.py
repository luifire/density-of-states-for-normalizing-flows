from plotting.utils import *


def _find_best_and_worst_images(interesting_classes, all_losses, all_loaders, amount_per_class):
    # stores the loss with images and class [loss] = (class, image)
    top_images = dict()
    bottom_images = dict()

    # for cls, losses, loader in interesting_losses:
    for cls in interesting_classes:
        losses = all_losses[cls][PD_IDX]
        loader = all_loaders[cls]

        # returns the indices that would sort this array
        sorted_loss_idcs = losses.argsort()
        # sort losses and images
        sorted_losses = losses[sorted_loss_idcs]
        sorted_images = loader.dataset.data[sorted_loss_idcs]

        for j in range(amount_per_class):
            # take the images from the start and the end of the (by loss) sorted array
            bottom_images[float(sorted_losses[j])] = (cls, sorted_images[j])
            top_images[float(sorted_losses[-1 - j])] = (cls, sorted_images[-1 - j])

    return top_images, bottom_images


def find_best_and_worst_images(interesting_classes, all_losses, all_loaders, amount_per_class):
    """Does _find_best_and_worst_images for every model"""
    best_imgs, worst_imgs = dict(), dict()
    for i, model in enumerate(all_losses):
        best_imgs[model], worst_imgs[model] = _find_best_and_worst_images(interesting_classes, all_losses[model],
                                                                 all_loaders, amount_per_class)

    return best_imgs, worst_imgs


def plot_best_or_worst_samples(model_name, interesting_classes, all_losses, all_loaders,
                               best_images, amount_per_class, cols, verbose=True):
    """model: [survae, nf, vae]
    best_images: if true we show best images, if false we show worst images
    interesting_classes: which classes you want to look at / compare (can be a single class)
    all_losses: precomputed losses
    all_loaders: loaders with images
    amount_per_class: how many images do you want per class
    cols: how many columns do you want your plot to have"""

    best_imgs, worst_imgs = _find_best_and_worst_images(interesting_classes, all_losses[model_name],
                                                                      all_loaders, amount_per_class)
    # needs to be put into a dict, because show outliers
    if best_images:
        images = best_imgs
    else:
        images = worst_imgs

    show_outliers(images, cols, model_name, desc=best_images, verbose=verbose)
