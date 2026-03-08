import time
import torch
from warnings import warn
import platform

from master.training.start_training_master import start_training_master
from master.config import *
from master.utils.utils import print_line, beep

def current_date_time():
    return time.strftime("%m.%d_%H-%M-%S")


def run_training(dataset, dimensions, variances, models, mode, epochs, train_samples, main_model,
                 num_blocks=3, additional_arguments='', wandb_name_ext=None, early_stopping=50, batch_size=32,
                 eval_increae_factor=1, uniform_noise=False, distance_evaluation=5):
    print(f'Start: {current_date_time()}')
    own_folder_name = f'inflation-{current_date_time()}-{dataset}-{mode}'
    if wandb_name_ext is None:
        wandb_name_ext = mode
    wandb_name_ext += f''

    eval_every = int(10 * eval_increae_factor)
    if eval_every == 0:
        eval_every = 1

    #beep(1000, 100)
    #warn('lowered learning rate')
    for var in variances:
        for dimension in dimensions:
            for model in models:
                ## --pdf_evaluation=10
                arguments = f'--root={MAIN_DIR} ' \
                            f'--dataset={dataset} --epochs={epochs} --eval_every={eval_every} ' \
                            f'--batch_size={batch_size} --own_folder={own_folder_name} ' \
                            f'--inflation --dimensions={dimension} --train_size={train_samples} ' \
                            f'--early_stopping={early_stopping} --manifold_evaluation={eval_every} ' \
                            f'--wandb_name_ext={wandb_name_ext} --sample_every=50 --distance_evaluation={distance_evaluation} ' \
                            f'--skip_scaling'

                if uniform_noise:
                    arguments += f' --uniform_noise={var}'
                elif var:
                    arguments += f' --var={var}'

                #warn('#############!!!!!!!!!!!!!!!!!!!!!!!!!!')
                #warn('changed lr')
                if main_model == 'nf':
                    arguments += f' --model=nf --flow={model} ' \
                                 f'--lr=1e-3 --num_blocks={num_blocks}'
                    # f'--lr=1e-3 ' \ --sample_every=None
                    # f'--early_stopping=50 --manifold_evaluation=10 ' \

                elif main_model == 'res_flow':
                    architecture_by_dim = {36: '16-16-16', 100: '13-12-13', 14**2: '13-12-13'}
                    arguments += f' --model=res_flow --actnorm True --wd 0 --nblocks={architecture_by_dim[dimension]}'

                if len(additional_arguments) > 0:
                    arguments += additional_arguments

                if mode == 'debug':
                    arguments += ' --debug'

                start_training_master(arguments.split(' '))
                torch.cuda.empty_cache()


def run_training_mnist(models, mode, epochs,
                       num_blocks=3, additional_arguments='', wandb_name_ext=None):

    print(f'Start: {current_date_time()}')
    own_folder_name = f'inflation-{current_date_time()}-mnist-{mode}'
    if wandb_name_ext is None:
        wandb_name_ext = mode

    for model in models:
        arguments = f'--root={MAIN_DIR} --model=nf --flow={model} ' \
                    f'--dataset=mnist --epochs={epochs} --eval_every=10 ' \
                    f'--dimensions={28**2} --batch_size=32 --own_folder={own_folder_name} ' \
                    f'--sample_every=1 --inflation --st_type=resnet ' \
                    f'--early_stopping=30 --eval_batch_size=64 ' \
                    f'--wandb_name_ext={wandb_name_ext} ' \
                    f'--lr=1e-4 --num_blocks={num_blocks} --dont_check --distance_evaluation=10'
                    #f'--uniform_noise={1e-2}' \

                    # f'--var={1e-2}  '\- --var={1e-3}
        ######### uniform_noise?!
        if len(additional_arguments) > 0:
            arguments += additional_arguments

        if platform.system() == 'Windows' and mode == 'debug':
            arguments += ' --dont_check'

        if mode == 'debug':
            arguments += ' --debug --dont_check'

        start_training_master(arguments.split(' '))
        torch.cuda.empty_cache()
