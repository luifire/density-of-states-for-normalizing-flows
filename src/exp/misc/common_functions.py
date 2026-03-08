import random
import numpy as np
import torch
from os import path
import pickle
from scipy.special import erf

from shared_functions import print_relevant_argument_info
from easy_access import EasyAccess


def load_config(checkpoint, data_path, verbose=True):
    # argument loading
    args = pickle.load(open(path.join(checkpoint, 'args.pickle'), 'rb'))

    # remove after this is standard
    args.overfit = False

    args.data_path = path.join(data_path, 'datasets')
    EasyAccess().args = args

    if verbose:
        print_relevant_argument_info(args)
    rename_config_stuff(args)
    return args


def rename_config_stuff(config):
    """config_entries = dir(config)
    if 'scale' not in config_entries:
        if 'var' in config_entries:
            config.scale = np.sqrt(config.var)"""

    if config.uniform_noise:
        if type(config.uniform_noise) == float:
            config.var = config.uniform_noise
            config.scale = np.sqrt(config.var)
            config.uniform_noise = True

    if 'tangent_scale' not in dir(config):
        config.tangent_scale = config.scale

def load_configs(checkpoints, data_path, verbose=True):
    """checkpoints contains a dict of {model_name: path to training info of that model}
     :returns a dict of {model_name: info to training of that model in that path}"""
    configs = dict()
    for key, path in checkpoints.items():
        configs[key] = load_config(path, data_path, verbose)
        #### remove when this is standard
        if 'eval_batch_size' not in dir(configs[key]):
            configs[key].eval_batch_size = 64

    # check if all training classes are equivalent
    training_class = -1
    for key in configs:
        current_training_class = configs[key].training_class
        if training_class == -1:
            training_class = current_training_class
        else:
            if current_training_class != training_class:
                raise Exception('Training classes don\'t match!')

    return configs, training_class


def digitize_image_values(img, original_resolution):
    """This produces an image in [0, 1] where the original image would have come from a resolution of original_resolution """
    return torch.floor(img * original_resolution) / original_resolution


# code from: https://docs.python.org/2/library/itertools.html#recipes
from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def create_theoretical_experts(all_models, expert_combiner, quantile_computation, psi_score):
    combo_scores = []
    for model_combo in powerset(all_models):
        if len(model_combo) == 0:
            continue
        combo_expert = expert_combiner(model_combo)
        quantile = quantile_computation(combo_expert)
        score = psi_score(combo_expert, quantile)
        combo_scores.append(((model_combo), score))
    sorted_scores = sorted(combo_scores, key=lambda model_score: (model_score[1], len(model_score[0])))
    sorted_scores.reverse()
    return sorted_scores


def remove_experts_without_effect(experts, round_digits=None):
    """we remove those models without effectcreate_theoretical_experts
    we keep single models for comparison.
    if round is not none, it is the position upon which we round (round=1, score=0.1234 => score => 0.1)"""
    previous_score = None
    indices_to_remove = []
    for item in experts:
        models, score = item
        # in case we store multiple scores, only take the first one
        if '__len__' in dir(score):
            score = score[0]

        if round_digits is not None:
            score = round(score, round_digits)
        # we remove those entries, where an additional model didn't bring any improvement
        if score == previous_score and not len(models) == 1:
            indices_to_remove.append(item)
        previous_score = score

    cleaned_scores = list(experts)
    for item in indices_to_remove:
        cleaned_scores.remove(item)
    return cleaned_scores


def pdf_N_plus_Uni(x, variance):
    return 1/2 * (erf((0.5 - x) / np.sqrt(variance * 2)) - erf((-0.5 - x) / np.sqrt(variance * 2)))
