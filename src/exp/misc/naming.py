########################
# This module is about class naming and coloring and so on
########################

# Plot Stuff
from matplotlib import pyplot as plt

# Master Stuff
from misc.constants import *
from master.config import *
from easy_access import EasyAccess


def class_to_name(cls):

    training_class = EasyAccess().args.training_class

    if cls == training_class:
        #return r'$\textbf{X}^t_' + f'{training_class}$'
        return f'{training_class} (test)'
    if cls == MAIN_TRAIN_CLS:
        return f'{training_class} (train)'

    if cls == STD_NORMAL_CLASS:
        return 'N(0,1)'
    if cls == AVERAGE_OUTLIERS:
        return 'Outlier Mean'

    return str(cls)


def get_legend_label(cls, val):
    #if cls == -1:
    #    assert False

    if type(val) is str:
        return f'{class_to_name(cls)} | {val}'
    else:
        return f'{class_to_name(cls)} | Mean: {val:.2f}'


def get_class_color(cls):
    training_class = EasyAccess().args.training_class
    # this assigns a specific color to every class
    if cls == training_class:
        return 'b'
    elif cls == MAIN_TRAIN_CLS:
        return 'r'
    elif cls == STD_NORMAL_CLASS:
        return 'g'
    else:
        return plt.get_cmap('tab10').colors[cls]


def get_model_color(model):
    model = model.lower()
    if 'realnvp' in model:
        return 'b'
    if 'realvp' in model:
        return 'y'
    if 'realnc' in model:
        return 'tab:cyan'
    if 'nice' in model:
        return 'm'
    if model == 'true':
        return 'k'
    """if model == 'RealNVP_uniform':
        return 'deepskyblue'
    if model == 'RealVP_uniform':
        return 'darkseagreen'
    if model == 'NICE_uniform':
        return 'hotpink'"""
    if model == 'n(0,1)':
        return 'tab:gray'
    if model == 'manifold':
        return 'm'
    if model == 'prior':
        return 'tab:gray'
    if model == 'resflow':
        return 'maroon'

    return None


def get_model_name(model, comparison_mode=False):
    """comparison_mode if true, we are comparing normal and uniform noise. This will be part of the name"""
    model = model.lower()
    if model.startswith('realnvp'):
        name = 'RealNVP'
    if model.startswith('realnc'):
        name = 'RealNC'
    if model.startswith('realvp'):
        name = 'RealVP'
    if model.startswith('nice'):
        name = 'NICE'

    if comparison_mode:
        if 'uniform' in model:
            name += ', uniform'
        else:
            name += ', normal'

    return name


def index_to_name(idx):
    if idx == CORRECTED_PD:
        return 'normed on manifold'
    if idx == PD_IDX:
        return 'densities'
    if idx == T_SCORE:
        return 'centered densities'

    return idx
