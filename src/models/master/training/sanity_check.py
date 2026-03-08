import wandb
from torchvision.utils import make_grid
import torchvision.utils as vutils
import torch
import numpy as np
from warnings import warn
from master.training.utils import simple_data_choice

from master.config import *
from easy_access import EasyAccess


def log_some_data(train_loader, eval_loader, others_eval_loaders):
    """" Here we write some data the same way they are feed into the model
    to wandb. This way making sure that the model sees the correct data. """

    def _log_images(loader, name):
        args = EasyAccess().args

        if int(np.sqrt(args.dimensions)) ** 2 != args.dimensions:
            warn('Not an image dimension. Can not do sanity check')
            return

        # collect the amount of images needed
        amount_of_batches = 1
        images = []
        #for i, (x, _) in enumerate(loader):
        for i, x in enumerate(loader):
            if i >= amount_of_batches:
                break
            images.append(x)
            
        x = torch.cat(images)

        # because eval batch size differs from training batch size
        amount = min(amount_of_batches * EasyAccess().args.batch_size, len(x))
        x = x[:amount]
        rows = int(amount ** 0.5)

        normalizer = 1 if args.dataset in simple_data_choice else 255.

        grid = make_grid(x / normalizer, nrow=rows)
        wand_img = wandb.Image(grid, caption=name)

        EasyAccess().wandb.log({name: wand_img}, step=0, commit=False)
        #vutils.save_image(x/.255, fp=f'C:\\test\\{name}.png', nrow=10)

    _log_images(train_loader, 'train_images')
    _log_images(eval_loader, 'test_images')

    for cls, loader in others_eval_loaders.items():
        _log_images(loader, str(cls) + '_test')

    # commit
    EasyAccess().wandb.log({}, step=0)
