from os import path
from pre_evaluation_main import PreEvaluationMain
from shared_functions import print_line


DEBUG = False
DATA = 'C:/Mega/trunk/master-thesis/data/'

### OLD
CHECKPOINTS = [ #'nf/RealNVP_blocks-6_mnist_cls-0/06.15_14-22-33_normal/',
                #'nf/RealNVP_blocks-6_mnist_cls-0/06.11_08-25-27_no_bias/',
                'nf/RealNVP_blocks-6_mnist_cls-0/06.24_13-35-41_new_loss_old',
                'survae/survae_steps-12_scales-2_pool-max_mnist_cls-0/06.11_14-20-54_normal/',
                'survae/survae_steps-12_scales-2_pool-max_mnist_cls-0/06.11_08-26-49_no_bias/',
                'survae/survae_steps-4_scales-5_pool-max_mnist_cls-0/06.11_09-35-05_hierarchy/',
                'survae/survae_steps-12_scales-2_pool-max_mnist_cls-0/06.10_20-27-32_new_loss',
                #'iwae/k-16_latent-30_mnist_cls-0/06.09_08-18-56_no_bias/',
                'iwae/k-16_latent-30_mnist_cls-0/06.10_20-27-53_with_bias/'
                ]

CHECKPOINTS = [
    #'nf/RealNVP_blocks-6_mnist_cls-0/08.13_09-15-57_exp_2/',
    #'nf/RealNVP_blocks-6_mnist_cls-0/08.13_09-16-28_exp_4/',
    #'nf/RealNVP_blocks-6_mnist_cls-0/08.13_09-30-22_exp_1.2/',
    #'nf/RealNVP_blocks-6_mnist_cls-0/08.13_09-33-39_exp_1.5/',
    #'conv/conv_net_mnist_cls-0/08.16_09-58-53'
    #'nf/NICE_blocks-6_mnist_cls-0/08.17_07-29-03_NICE/',
    #'nf/NICE_blocks-6_fashionmnist_cls--1/08.19_10-54-04_NICE',
    'nf/RealNVP_blocks-6_fashionmnist_cls--1/08.19_10-55-52_normal',
    'nf/RealNVP_blocks-6_fashionmnist_cls--1/08.19_10-55-57_new_loss'
]

for i, checkpoint in enumerate(CHECKPOINTS):
    print_line()
    print_line()
    print(f'{i+1} / {len(CHECKPOINTS)}')

    checkpoint_path = path.join(DATA, 'checkpoints', checkpoint + '/')
    PreEvaluationMain(checkpoint_path, DATA, DEBUG).run()
