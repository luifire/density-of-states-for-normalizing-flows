from master.utils.utils import print_line

from master.train_settings.multi_training import run_training, current_date_time

if __name__ == '__main__':
    # 5**2 doesn't work
    models = ['RealNVP', 'NICE']
    dimensions = [6**2, 10**2, 14**2]

    run_training('gauss', dimensions=dimensions, variances=[1], models=models,
                 epochs=100, mode='exp', train_samples=1000, num_blocks=3)

    print_line()
    print('Finished successfully')
    print(f'End: {current_date_time()}')
    print_line()
    print_line()
