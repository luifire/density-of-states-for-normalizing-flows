"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluation
import projection as proj
import net_plotter
from net_plotter import *
import plot_2D
import plot_1D
from model_loader import *
import scheduler
import mpi4pytorch as mpi

from h52vtp import run_h52vtp
from utils import *

from master.training.utils import get_loss_fn
from master.data.data_master import get_data


def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    #surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    #if args.y:
    #    surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"

def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        os.remove(surf_file)
        if False:
            f = h5py.File(surf_file, 'r')
            if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
                f.close()
                print ("%s is already set up" % surf_file)
                return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args, data_shape):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    """criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()
    """

    criterion = get_loss_fn(net.config, data_shape)

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord, args=args)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord, args=args)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            if count % 1000 == 0:
                f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    f.flush()
    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    if max(inds_nums) - len(inds) != 0:  # David
        for _ in range(max(inds_nums) - len(inds)):
            losses = mpi.reduce_max(comm, losses)
            accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    dim = 100
    log_var = -4

    flow = 'NICE'
    flow = 'RealVP'
    flow = 'RealNVP'

    checkpoint = f'inflation-04.04_11-55-40-smooth_step-exp/{flow}_blocks-3_smooth_step/dim{dim}_var1e{log_var}.0'
    #checkpoint = f'inflation-04.04_11-55-40-smooth_step-exp/{flow}_2d_smooth_step/dim2_var1e{log_var}.0'
    #checkpoint = f'no_rotation/inflation-03.28_13-53-14-smooth_step-exp_--no_rotation/{flow}_blocks-3_smooth_step/dim{dim}_var1e{log_var}.0'

    #checkpoint = f'inflation-05.11_13-38-48-smooth_step-exp_--mighty_prior-batch=32-samples=2000/{flow}_blocks-3_smooth_step/dim{dim}_var1e{log_var}.0'
    #checkpoint = 'stays_inflation-04.28_13-55-35-smooth_step-exp_--discard_near_0_z=0.0001--mighty_prior-batch=32-samples=1000' \
    #             f'/{flow}_blocks-3_smooth_step/dim{dim}_var1e{log_var}.0'

    granularity = 100
    #granularity = 20
    if 'dim36' in checkpoint or 'dim2' in checkpoint:
        range = 2.5
    elif 'dim100' in checkpoint:
        range = 2.5
    else:
        range = 2.5

    norm = 'filter' # filter | layer | weight
    command_line = f'--cuda --x=-{range}:{range}:{granularity} --y=-{range}:{range}:{granularity} ' \
                   f'--dataset=smooth_step --data_path=F:/master_data/checkpoints/ ' \
                   f'--dir_type weights --plot --vmax_factor=10 --xnorm={norm} --ynorm={norm} ' \
                   #f'--weight_filter=per_st_net '
                   #'--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot '

    data_mode = 'sampled_manifold'
    #data_mode = 'sampled'

    command_line += f'--checkpoint={checkpoint} --data_mode={data_mode}'
    #command_line += ' --filter_scaling'  # rather not, makes things very smooth

    args = surface_plot_arguments(command_line)

    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    torch.manual_seed(51090)
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = load_model_for_loss_plot(args)
    w = net_plotter.get_weights(net, args) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    print('Warning: might not override old surface file')
    dir_file = name_direction_file(args, net.config) # name the direction file
    if rank == 0:
        setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    #if rank == 0 and args.dataset == 'cifar10':
    #    torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

    #mpi.barrier(comm)

    net.config.batch_size = 1024
    trainloader, testloader, data_shape, _ = get_data(args=net.config, main_class=-1, shuffle_train=True,
                                                   verbose=True, data_mode=args.data_mode)
    """dataset = load_dataset(net.config, dataset_size=1000, data_mode=args.data_mode)
    train_loader = DataLoader(dataset.train, batch_size=args.batch_size, shuffle=shuffle_train,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    """
    """trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
                                args.batch_size, args.threads, args.raw_data,
                                args.data_split, args.split_idx,
                                args.trainloader, args.testloader)
    """
    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    net = net.cuda()
    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args, data_shape)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        #args.vmin, args.vmax, args.vlevel = None, None, None  # David

        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
        elif args.y:
            vmax, Z = plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vmax_factor, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)

    #########
    h52vtp_args = f'--surf_file={surf_file} --surf_name train_loss --zmax={min(int(Z.max()), 100)}'
    run_h52vtp(h52vtp_args.split(' '))