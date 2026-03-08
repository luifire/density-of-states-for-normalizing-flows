import pickle
import os
import sys
import time
import numpy as np
import platform
from pathlib import Path

from master.config import *
from misc.constants import LOG_NO_INFLATION_VAR
from shared_functions import print_line
from easy_access import EasyAccess


def get_bool_from_args(name):
    args = EasyAccess().args
    if name in dir(args):
        return args.__getattribute__(name)
    return False


def print_all_arguments(arguments=None):
    """Prints all command line arguments"""
    print_line()
    if arguments is None:
        arguments = sys.argv

    print(' '.join(arguments))
    print_line()


def create_folders(args, model_id):
    own_folder = args.own_folder

    """Creates needed folder for training"""
    if own_folder != '':
        last_folders_name = ''
    else:
        last_folders_name = time.strftime("%m.%d_%H-%M-%S")

    if args.inflation:
        var = args.uniform_noise if args.uniform_noise else args.var
        if var is None or np.isclose(np.log10(var), LOG_NO_INFLATION_VAR):
            var = 0

        last_folders_name += f'dim{args.dimensions}_var1e{np.log10(var):.2}'
        
    if args.wandb_name_ext != '':
        own_folder += f'_{args.wandb_name_ext}'

    if args.debug:
        last_folders_name += '_debug'

    if '~' in args.root:
        args.root = args.root.replace('~', str(Path.home()))

    args.log_path = os.path.join(args.root, 'checkpoints', own_folder, model_id, last_folders_name)
    args.sample_root = os.path.join(args.log_path, 'samples')
    args.check_path = os.path.join(args.log_path, 'check')
    args.intermediate_checks = os.path.join(args.root, 'intermediate', model_id)

    # Create log folder
    os.makedirs(args.log_path, exist_ok=True)
    print("Storing logs in:", args.log_path)
    os.makedirs(args.log_path + '/manifold', exist_ok=True)

    os.makedirs(args.log_path + '/dist_change', exist_ok=True)

    # Create check folder
    os.makedirs(args.check_path, exist_ok=True)
    print("Storing checkpoints in:", args.check_path)

    # Create sample folder
    if args.num_samples > 0:
        os.makedirs(args.sample_root, exist_ok=True)

    # Create Intermediate check folder
    os.makedirs(args.intermediate_checks, exist_ok=True)


def beep(frequency=1000, duration=100):
    if platform.system() == 'Windows':
        import winsound
        winsound.Beep(frequency, duration)
    else:
        print('beeeeeep')
