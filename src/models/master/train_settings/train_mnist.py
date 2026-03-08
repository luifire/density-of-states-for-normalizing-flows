import numpy as np

from misc.constants import *
from master.utils.utils import print_line, beep
from master.train_settings.multi_training import run_training_mnist, current_date_time


if __name__ == '__main__':
    MODE = 'exp'
    if MODE == 'exp':
        epochs = 200
        num_blocks = 3
    elif MODE == 'debug':
        epochs = 1
        num_blocks = 1
        dimensions = [6**2, 8**2]
        variances = [1e-1, 1e-0]

    #models = ['RealNVP', 'NICE']
    #models = ['NICE']
    models = ['RealVP']
    #log_variances = ALL_LOG_VARIANCES
    #log_variances = [-6, -4, -2, -1, -0.5, 0, 0.25, LOG_NO_INFLATION_VAR]
    #log_variances = [LOG_NO_INFLATION_VAR]
    #variances = 10.0 ** np.array(log_variances)

    # main training
    additional_arguments = ''

    #import torch
    #torch.autograd.set_detect_anomaly(True)

    beep(1000, 100)
    #additional_arguments += '--full_noise'
    #additional_arguments += ' --dont_check'
    #additional_arguments += ' --dont_check --debug'
    #additional_arguments += ' --mighty_prior'
    #additional_arguments += ' --pre_inflation_ilogits'
    additional_arguments += f' --var={1e-4}'
    #additional_arguments += f' --discard_near_0_z={1e-6}'

    run_training_mnist(models=models,
                       mode=MODE, num_blocks=num_blocks,
                       epochs=epochs, additional_arguments=additional_arguments)

    print_line()
    print('Finished successfully')
    print(f'End: {current_date_time()}')
    print_line()
    print_line()
