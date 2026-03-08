import numpy as np
import torch

from master.data.data_helper import *
from misc.constants import *


def get_full_space_data(dim, dataset_name, dataset):
    if dataset_name == 'smooth_step':
        #start, end = -5.5, 5.5
        #density = 0.05
        #density = 2
        start, end = -1, 1
        density = 0.005

    elif dataset_name == 'line':
        start, end = -1, 1
        density = 0.005

    a = torch.arange(start, end, density).float()
    data = torch.cartesian_prod(a, a)
    data = data.numpy()
    if dim == 2:
        llh = dataset.log_pdf(data)
        # should look like: [SAMPLE_COUNT, 1, 1, 2]
        data_store = data[:, None, None, :]
    else:

        # only the first two dimensions are filled up, the rest will stay 0
        data_store = torch.zeros((len(data), dim))
        data_store[:, 0] = data[0]
        data_store[:, 1] = data[1]
        data_store = data
        llh = dataset.log_pdf(data_store)

    return torch.from_numpy(data_store), llh
