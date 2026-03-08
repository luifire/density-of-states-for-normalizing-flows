from warnings import warn
import numpy as np

from master.config import *
from master.utils.utils import print_line, beep

from master.train_settings.multi_training import current_date_time
from master.training.start_training_master import start_training_master
from misc.constants import *
from master.train_settings.utils import compute_required_steps

if __name__ == '__main__':
    train_samples = 1000
    batch_size = 32

    epochs, early_stop, eval_increase_factor = compute_required_steps(batch_size, train_samples, default_early_stop=100)

    flows = ['RealNVP', 'NICE']
    flows = ['RealNVP']
    flows = ['NICE']

    uniform_noise = True
    datasets = ['sphere', 'line', 'smooth_step']
    datasets = ['sphere']
    #datasets = ['line']
    #datasets = ['smooth_step']
    #datasets = ['gauss']
    log_variances = ALL_LOG_VARIANCES
    #log_variances = [-6, -4, -2, -1, None]
    log_variances = [-6, -4, -2, -1]
    #log_variances = [None]
    log_variances = [-6, -4]
    #log_variances = [-2, -1]

    additional_arguments = ''
    #additional_arguments += ' --adjust_prior_var'
    #additional_arguments += ' --new_loss'
    #additional_arguments += ' --dont_check'
    #additional_arguments += ' --debug'
    #additional_arguments = f' --trainable_scaling=1'
    #additional_arguments += ' --no_rotation'
    #additional_arguments += ' --adjust_scaling_weights'

    dataset_parameters = {'sphere': MISES_ARGS,
                          'smooth_step': SMOOTH_STEP_ARGS,
                          'line': '', 'gauss': ''}
    #warn('Zu viel manifold logging')
    beep(1000, 100)
    wandb_name_ext = ''
    wandb_name_ext += additional_arguments.replace(' ', '')
    wandb_name_ext += f'-samples={train_samples}'

    np_var = np.array(log_variances)
    variances = list(10.0 ** np_var[np_var != None])
    if None in np_var:
        variances.append(None)

    for dataset in datasets:
        own_folder_name = f'inflation-{current_date_time()}-2d-{dataset}'
        for var in variances:
            for flow in flows:
                # batch size and blocks and lr
                # --sample_every=None --early_stopping=100
                arguments = f'--model=2d_nf --root={MAIN_DIR}' \
                            f' --dataset={dataset} --epochs={epochs} --eval_every={int(eval_increase_factor * 10)}' \
                            f' --lr=1e-3 --num_couplings=10 --flow={flow}' \
                            f' --batch_size={batch_size} --own_folder={own_folder_name}' \
                            f' --inflation --dimensions=2 --train_size={train_samples}' \
                            f' --wandb_name_ext=2d_nf_{wandb_name_ext} --list_dimension --early_stopping={early_stop}' \
                            f' --manifold_evaluation=10 --distance_evaluation=10' \
                            + additional_arguments

                if uniform_noise:
                    arguments += f' --uniform_noise={var}'
                elif var:
                    arguments += f' --var={var}'

                arguments += dataset_parameters[dataset]

                start_training_master(arguments.split(' '))

    print_line()
    print('Finished successfully')
    print_line()
