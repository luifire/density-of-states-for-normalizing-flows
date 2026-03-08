from master.config import *
from master.utils.utils import print_line

from master.train_settings.multi_training import current_date_time
from master.training.start_training_master import start_training_master


if __name__ == '__main__':
    # 5**2 doesn't work
    #models = ['RealNVP', 'NICE']
    models = ['NICE']

    for model in models:
        own_folder_name = f'inflation-{current_date_time()}-small_gauss'

        # batch size and blocks and lr
        arguments = f'--model=nf --root={MAIN_DIR} --flow={model} --st_type=fully --latent_dim=8' \
                    f' --dataset=gauss --epochs=1000 --eval_every=10 --sample_every=20' \
                    f' --lr=1e-2' \
                    f' --num_blocks=20 --batch_size=256 --own_folder={own_folder_name}' \
                    f' --inflation --var=1 --dimensions=4 --train_size=3000' \
                    f' --wandb_name_ext=small_gauss --debug --skip_scaling --no_scaling_layer'  # --list_dimension'
                    # f' --early_stopping=best_of_30' \
        start_training_master(arguments.split(' '))

    print_line()
    print('Finished successfully')
    print_line()
