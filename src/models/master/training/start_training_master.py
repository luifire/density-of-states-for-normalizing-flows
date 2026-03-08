# General
import os

# Master
from survae.experiments.image.utils import set_seeds
from master.utils.arguments import load_args, save_args
from master.utils.utils import print_line, create_folders
from master.utils.utils import print_all_arguments
from master.data.inflation import NormalInflation, UniformInflation

# Data
from master.data.data_master import get_data, get_data_id, create_evaluation_loaders
# Exp
from master.training.training_master import FlowExperiment_Master
# Rest
from master.training.utils import *
from easy_access import EasyAccess
from shared_functions import print_relevant_argument_info
from inflation.jupyter_helper import load_dataset

##################
# (Split) means it is different for NF and SurVAE - (Common) is the opposite
##################


def start_training_master(arguments):
    """starts the main training"""

    ##############
    # Arg Parsing (Split)
    print_all_arguments(arguments)
    args = load_args(arguments)

    ##############
    # Initialize Stuff (Common)
    set_seeds(args.seed)

    ##############
    # Data Loading (Common)
    args.data_path = os.path.join(args.root, 'datasets')

    train_loader, eval_loader, data_shape, dataset_creator = get_data(args, args.training_class)
    args.data_shape = data_shape
    print(f'datashape is {data_shape}')

    other_classes_loaders = create_evaluation_loaders(args, use_train_loaders=False)

    ##############
    # Manifold evaluation data
    manifold_evaluation_data = grid_evaluation_data = None
    if args.manifold_evaluation is not None:
        manifold_evaluation_data = load_dataset(args, dataset_size=None, data_mode='manifold_grid')
    if args.pdf_evaluation is not None:
        grid_evaluation_data = load_dataset(args, dataset_size=None, data_mode='full_grid_fine')

    ##############
    # Model Loading (Split)
    model, model_id = get_model(args, data_shape)

    ##############
    ## Loss Function (Split)
    loss_fn = get_loss_fn(args, data_shape)

    ##############
    ## initalize model (Common)
    initialize_model(model, args)

    ##############
    # Optimizer (Split)
    optimizer, scheduler_iter, scheduler_epoch = get_optimizer(args, model, loss_fn)

    ##############
    # Folder (Common)
    create_folders(args, model_id)

    ##############
    # Initialize W&B (Common)
    wandb_run = get_logging(args, mock_wandb=args.debug)
    #wandb_run = get_logging(args, mock_wandb=True)

    ##############
    # Inflation (Common)
    inflation = None
    if args.var:
        inflation = NormalInflation(args.scale, args.tangent_scale, data_shape, dataset_creator)
    if args.uniform_noise:
        inflation = UniformInflation(args.uniform_noise)

    ##############

    if args.debug:
        pass
        #args.epochs = 10

    print_relevant_argument_info(args)

    ## Training (Common)
    exp = FlowExperiment_Master(args=args,
                                epochs=args.epochs,
                                data_id=get_data_id(args),
                                model_id=model_id,
                                train_loader=train_loader,
                                eval_loader=eval_loader,
                                other_classes_loaders=other_classes_loaders,
                                model=model,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                scheduler_iter=scheduler_iter,
                                scheduler_epoch=scheduler_epoch,
                                logging=wandb_run,
                                inflater=inflation,
                                manifold_evaluation_data=manifold_evaluation_data,
                                grid_evaluation_data=grid_evaluation_data)

    # load weights from given path
    if args.load_checkpoint is not None:
        checkpoint_path = path.join(args.root, 'checkpoints', args.load_checkpoint, 'check')
        exp.checkpoint_load(checkpoint_path)

    save_args(exp.log_path, args)

    print_line()
    exp.run()


if __name__ == '__main__':
    start_training_master(None)

