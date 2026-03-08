import numpy as np

from misc.constants import *
from master.utils.utils import print_line, beep
from master.train_settings.multi_training import run_training, current_date_time
from master.train_settings.utils import compute_required_steps

if __name__ == '__main__':
    MODE = 'exp'
    if MODE == 'exp':
        epochs = 1000
        num_blocks = 3
    elif MODE == 'debug':
        epochs = 100
        num_blocks = 3

    main_model = 'nf'
    models = ['NICE', 'RealNVP', 'RealVP']
    datasets = ['sphere', 'line', 'smooth_step']
    dimensions = [36, 100, 196]

    batch_size = 32
    train_samples = 1000
    #train_samples = 20000

    epochs, early_stop, eval_increase_factor = compute_required_steps(batch_size, train_samples)
    #epochs = early_stop = 2000

    models = ['NICE']
    #models = ['RealVP']
    models = ['RealNVP']
    #dimensions = [36]
    dimensions = [100]
    #dimensions = [196]
    #dimensions = [36, 100]

    log_variances = ALL_LOG_VARIANCES
    #log_variances = [-6, -4, -2, -1, None]
    #log_variances = [-2, -1]
    #log_variances = [-4, -2, -1, None]
    #log_variances = [-2, -1]
    log_variances = [-2, -4, -6, -1, None]

    #log_variances = [-6]
    #log_variances = [-4]
    log_variances = [-2]
    #log_variances = [-1]
    #log_variances = [None]
    #log_variances = [-6, -4,]
    #log_variances = [-2, -1]
    #log_variances = [None]

    datasets = ['smooth_step']
    #datasets = ['line']
    #datasets = ['sphere']

    uniform_noise = False

    # stupid no-variance
    np_var = np.array(log_variances)
    variances = list(10.0 ** np_var[np_var != None])
    if None in np_var:
        variances.append(None)

    #main_model = 'res_flow'
    #SMOOTH_STEP_ARGS = '--data_start=-50 --data_end=50 --data_offset=1 --data_range=2 --data_growth=2'

    wandb_name_ext = ''
    #wandb_name_ext = 'weight_change[!]'  # no spaces!
    #wandb_name_ext = 'samples=20k'  # no spaces!
    #wandb_name_ext = '1K_epochs'
    #trainable_scales = 1
    #wandb_name_ext = 'go_down_inflation_0.7'
    #wandb_name_ext = f'trainable_scaling={trainable_scales}'
    additional_arguments = ''

    #additional_arguments = ' --num_scales=5'

    #additional_arguments = f' --trainable_scaling=1'
    #additional_arguments += ' --no_rotation'
    # additional_arguments += ' --st_type=fully --num_scales=2 --skip_scaling'
    # additional_arguments += ' --no_scaling_layer'
    #additional_arguments += ' --adjust_scaling_weights'
    # additional_arguments += ' --dont_check --debug'
    # additional_arguments += ' --adjust_prior_var'
    #additional_arguments += ' --pre_inflation_ilogits'
    #additional_arguments += ' --dont_check'
    #additional_arguments += f' --discard_near_0_z={1e-4}'
    #additional_arguments += ' --pdf_evaluation=8'
    #additional_arguments += ' --mighty_prior'
    #additional_arguments += f' --tangent_noise_var={1e-1}'

    #import torch
    #torch.autograd.set_detect_anomaly(True)

    if uniform_noise:
        wandb_name_ext += 'uniform_noise_'

    wandb_name_ext += additional_arguments.replace(' ', '')

    if MODE == 'debug':
        additional_arguments += ' --dont_check'

    #additional_arguments += ' --dont_check'
    # beep(1000, 500)

    beep(1000, 500)
    distance_evaluation = 1
    #early_stop = 'None'
    #epochs = 1000
    additional_arguments += ' --dont_check'

    if 'dont_check' in additional_arguments:
        beep(1000, 1500)

    #beep(1000, 1500)
    #early_stop = 50

    # main training
    for dataset in datasets:
        beep(1000, 100)

        dataset_specific_arguments = additional_arguments
        if dataset == 'line':
            ...
        elif dataset == 'sphere':
            dataset_specific_arguments += MISES_ARGS
        elif dataset == 'smooth_step':
            dataset_specific_arguments += SMOOTH_STEP_ARGS

        run_training(dataset, epochs=epochs, dimensions=dimensions, variances=variances, models=models,
                     mode=MODE, train_samples=train_samples, num_blocks=num_blocks,
                     additional_arguments=dataset_specific_arguments, main_model=main_model, wandb_name_ext=wandb_name_ext,
                     early_stopping=early_stop, batch_size=batch_size, eval_increae_factor=eval_increase_factor,
                     uniform_noise=uniform_noise, distance_evaluation=1)

    print_line()
    print('Finished successfully')
    print(f'End: {current_date_time()}')
    print_line()
    print_line()
