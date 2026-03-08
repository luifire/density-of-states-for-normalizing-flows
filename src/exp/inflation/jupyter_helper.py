import os
from os import path, getcwd
import warnings
import torch
import copy
import numpy as np
from IPython import get_ipython

from misc.constants import *
from misc.common_functions import load_config
from probability_functions import *

from master.data.sphere_data import VonMises
from master.data.line_data import LineData
from master.data.gauss_data import GaussData
from master.data.smoothstep_data import SmoothstepData
from inflation.evaluate_model import evaluate_model
from precomputation.initialization import *
from res_flow.resflow import ResidualFlow


def get_all_checkpoints(root):
    _, models, _ = list(os.walk(root))[0]

    all_checkpoints = []
    for model in models:
        model_path = path.join(root, model)
        _, model_checkpoints, _ = list(os.walk(model_path))[0]
        all_checkpoints += [path.join(model_path, checkpoint) for checkpoint in model_checkpoints]
    return all_checkpoints


def sort_all_checkpoints(all_checkpoints, data_path):
    special_models = ['ResFlow', 'NICE_uniform', 'RealNVP_uniform', 'RealVP_uniform']
    all_models = {'NICE': dict(), 'RealNVP': dict(), 'RealVP': dict(), 'ResFlow': dict()}

    for special_model in special_models:
        all_models[special_model] = dict()

    for checkpoint in all_checkpoints:
        config = load_config(checkpoint, data_path, verbose=False)
        if 'NICE' in checkpoint:
            model = 'NICE'
        elif 'RealVP' in checkpoint:
            model = 'RealVP'
        elif 'RealNVP' in checkpoint:
            model = 'RealNVP'
        elif 'res_flow' in checkpoint:
            model = 'ResFlow'

        # add the uniform extension
        if 'uniform' in checkpoint:
            model += '_uniform'

        # print(config.var)
        all_models[model].setdefault(config.dimensions, dict())

        if config.var is None or np.log10(config.var) == LOG_NO_INFLATION_VAR:
            config.var = 0
        all_models[model][config.dimensions][config.var] = checkpoint

    # remove special model if they are not present
    for special_model in special_models:
        if special_model in all_checkpoints and len(all_checkpoints[special_model]) > 0:
            del all_checkpoints[special_model]

    return all_models


def get_all_dimensions_and_variances(all_models):
    dimensions = np.sort(list(all_models['NICE'].keys()))
    variances = np.sort(list(all_models['NICE'][list(dimensions)[0]]))
    # we ignore variance bigger 1. as the noise is to much for this
    return dimensions, variances[variances <= 1e-1]
    #return dimensions, variances


def setdefault_multiple_depth(dictionary, a, b, c):
    dictionary.setdefault(a, dict()),
    dictionary[a].setdefault(b, dict()),
    dictionary[a][b].setdefault(c, dict())


def add_gauss_model(all_models, dimension, var, size, t_score_divide_by_var):
    setdefault_multiple_depth(all_models, 'N(0,1)', dimension, var)
    data, log_prob = GaussData(dimension=dimension, train_size=10).create_test_data(size=size)
    all_models['N(0,1)'][dimension][var][PD_IDX] = log_prob
    all_models['N(0,1)'][dimension][var][T_SCORE] = (log_prob - log_prob.mean()) / (
        log_prob.var() if t_score_divide_by_var else 1)
    all_models['N(0,1)'][dimension][var][Z_IDX] = data


def create_corrected_pds(model, true_model, var):
    if 'ambient' in model.keys():
        model['ambient'][CORRECTED_PD], model['ambient'][CORRECTION_TERM] = normalize_q_with_p(model['ambient'][PD_IDX],
                                                                                   true_model['ambient'][PD_IDX])
        if var is None or var == 0:
            model['ambient'][PD_IDX] = model['ambient'][CORRECTED_PD]

    if 'manifold_grid' in model.keys():
        # this is a 1D grid, we can normalize it by itself
        model['manifold_grid'][CORRECTED_PD], model['manifold_grid'][CORRECTION_TERM] = \
            normalize_grid(model['manifold_grid'][PD_IDX], true_model['manifold_grid']['dataset'].data[INFLATED_MANIFOLD_DATA])

    if 'manifold' in model.keys():
        model['manifold'][CORRECTED_PD], model['manifold'][CORRECTION_TERM] = \
            normalize_q_with_p(model['manifold'][PD_IDX], true_model['manifold'][PD_IDX])

    warnings.warn('This should be reconsidered, when X is not bounded! (Like z ~ N)')

    # create the corrected pd for grids if their sampled part exists
    #if 'manifold_grid' in model.keys() and 'manifold' in model.keys():
    #    model['manifold_grid'][CORRECTED_PD] = model['manifold_grid'][PD_IDX] - model['manifold'][CORRECTION_TERM]

    if 'full_grid' in model.keys():
        # if a sampled correction term was computed
        if 'ambient' in model.keys():
            model['full_grid'][CORRECTED_PD] = model['full_grid'][PD_IDX] - model['ambient'][CORRECTION_TERM]


def load_dataset(config, dataset_size=None, data_mode='sampled'):
    kwargs = {'normal_noise_scale': config.scale, 'tangent_noise_scale': config.tangent_scale, 'sample_seed': 42,
              'size': dataset_size, 'dimension': config.dimensions, 'data_mode': data_mode}

    if config.var is None or np.log10(config.var) == LOG_NO_INFLATION_VAR:
        kwargs['normal_noise_scale'] = None
        kwargs['tangent_noise_scale'] = None
    kwargs['uniform_noise'] = config.uniform_noise is not None
    if config.no_rotation:
        kwargs['rotation_seed'] = None

    if config.dataset == 'sphere':
        return VonMises(sphere_radius=config.sphere_radius, mises_kappa=config.mises_kappa, **kwargs)
    elif config.dataset == 'line':
        return LineData(line_length=config.line_length, **kwargs)
    elif config.dataset == 'smooth_step':
        return SmoothstepData(start=config.data_start, end=config.data_end, offset=config.data_offset,
                              Range=config.data_range, growth=config.data_growth, **kwargs)
    elif config.dataset == 'gauss':
        return GaussData(**kwargs)


def get_inflation_fig_save_path(save_path, dataset_name, dimensions, extension=''):
    if save_path is not None:
        if len(dimensions) == 1:
            dims = f'{dimensions[0]}D'
        elif len(dimensions) >= 3:
            dims = 'high D'
        else:
            dims = ''
        dir = f'inflation/{dims}/{extension}/'
        os.makedirs(dir, exist_ok=True)
        return f'{dir}/{save_path}_{dataset_name}'


def save_inflation_fig(fig, save_path, dataset_name, dimensions, extension=''):
    if save_path is not None:
        save_path = get_inflation_fig_save_path(save_path, dataset_name, dimensions, extension)
        #fig.savefig(f'{save_path}.png', bbox_inches='tight')
        fig.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')


def get_smallest_quantile_of_model(model, alpha):
    bottom, top = float('inf'), float('-inf')
    for model, data in model.items():
        if model == 'N(0,1)':
            continue
        sorted_values = np.sort(list(data))
        N = len(sorted_values)

        bottom = min(bottom, sorted_values[int(alpha * N)])
        top = max(top, sorted_values[int((1 - alpha) * N)])
        # print(model, top)
    return bottom, top


def quantile_finder(data, alpha):
    sorted = np.sort(data)
    N = len(data)
    return sorted[int(N * alpha)], sorted[int(N * (1-alpha)) - 1]


def dataset_to_name(dataset):
    if dataset in ['sphere', 'mises']:
        return 'von Mises'
    elif dataset == 'line':
        return 'Uniform'
    elif dataset == 'smooth_step':
        return 'Smooth Step'
    elif dataset == 'gauss':
        return 'Gauss'
    else:
        return dataset

def dataset_to_name_abbrev(dataset):
    if dataset in ['sphere', 'mises']:
        return 'von Mises (Mises)'
    elif dataset == 'line':
        return 'Uniform (U)'
    elif dataset == 'smooth_step':
        return 'Smooth Step (SmSt)'
    else:
        return dataset


def _load_data_by_experiment(experiment, config, eval_dataset_size):
    if experiment == 'inflation':
        sampled_dataset = load_dataset(config, eval_dataset_size, data_mode='sampled')
        sampled_manifold = load_dataset(config, eval_dataset_size//5, data_mode='sampled_manifold')
        grid_manifold_dataset = load_dataset(config, data_mode='manifold_grid')

        key_to_dataset = {'ambient': sampled_dataset, 'manifold': sampled_manifold,
                          'manifold_grid': grid_manifold_dataset}

    elif experiment == 'latent_transformation':
        sampled_dataset = load_dataset(config, eval_dataset_size, data_mode='sampled')
        grid_manifold_dataset = load_dataset(config, data_mode='manifold_grid')

        key_to_dataset = {'ambient': sampled_dataset, 'manifold_grid': grid_manifold_dataset}
    elif experiment == 'full_space':
        full_grid = load_dataset(config, data_mode='full_grid')
        sampled_dataset = load_dataset(config, data_mode='sampled', dataset_size=500)

        key_to_dataset = {'ambient': sampled_dataset, 'full_grid': full_grid}

    true_model = dict()
    for key, dataset in key_to_dataset.items():
        true_model[key] = dict()
        true_model[key]['dataset'] = dataset
        true_model[key][EMBEDDED_DATA] = dataset.data[EMBEDDED_DATA]
        true_model[key][NONE_EMBEDDED_DATA] = dataset.data[NONE_EMBEDDED_DATA]
        # we do not always use all datasets, thus they are considered separately
        # also we are interested in different densities
        if key == 'ambient':
            true_model['ambient'][PD_IDX] = key_to_dataset['ambient'].densities[FULL_INFLATED_LOG_PD]
        elif key == 'manifold':
            true_model['manifold'][PD_IDX] = key_to_dataset['manifold'].densities[INFLATED_MANIFOLD_LOG_PD]
        elif key == 'manifold_grid':
            true_model['manifold_grid'][PD_IDX] = key_to_dataset['manifold_grid'].densities[INFLATED_MANIFOLD_LOG_PD]
        elif key == 'full_grid':
            true_model['full_grid'][PD_IDX] = key_to_dataset['full_grid'].densities[FULL_INFLATED_LOG_PD]

    for key in true_model.keys():
        true_model[key][CORRECTED_PD] = true_model[key][PD_IDX]

    for key in true_model.keys():
        if key not in ['manifold_grid', 'full_grid']:
            true_model[key][T_SCORE] = true_model[key][CORRECTED_PD] - true_model[key][CORRECTED_PD].mean()

    return true_model


def load_model_from_checkpoint(checkpoint, data_path):
    config = load_config(checkpoint, data_path, verbose=False)
    model = load_model(checkpoint, verbose=False)
    model.config = config

    rename_model_stuff(model)

    model.eval()
    return model


def rename_model_stuff(model):
    content = dir(model)
    if 'implementation' not in content:
        model.implementation = 'nice' if model.nice_implementation else 'RealNVP'
    if 'nice_implementation' not in content:
        model.nice_implementation = model.implementation == 'nice'


def create_stored_experiment_path(experiment, data_path, config, data_extension):
    var = '-inf' if config.var is None else np.log10(config.var)
    return path.join(data_path, 'preloaded_experiments', f'{experiment}-{config.dataset}-{data_extension}',
                     f'dim{config.dimensions}-var{var}.exp')


def was_experiment_loaded(experiment, data_path, config, data_extension):

    # config mit abspeichern und gegen checken
    experiment_path = create_stored_experiment_path(experiment, data_path, config, data_extension)
    if path.exists(experiment_path):
        dump = pickle.load(open(experiment_path, 'rb'))
        if dump['config'] == config:
            return True

    return False


def load_experiment_results(experiment, data_path, config, data_extension, all_models):
    experiment_path = create_stored_experiment_path(experiment, data_path, config, data_extension)
    dump = pickle.load(open(experiment_path, 'rb'))

    var, dimension = config.var, config.dimensions
    if var is None or np.isclose(np.log10(var), -1.23):
        var = 0

    for model_name in dump.keys():
        if model_name == 'config':
            continue
        setdefault_multiple_depth(all_models, model_name, dimension, var)
        all_models[model_name][dimension][var] = dump[model_name]


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if not path.exists(directory):
        os.makedirs(directory)


def save_experiment_results(experiment, data_path, config, data_extension, all_models):
    var, dimension = config.var, config.dimensions
    if var is None or np.isclose(np.log10(var), -1.23):
        var = 0

    experiment_path = create_stored_experiment_path(experiment, data_path, config, data_extension)

    stored_dict = {model_name: all_models[model_name][dimension][var] for model_name in all_models.keys()}
    stored_dict['config'] = config
    ensure_dir_exists(experiment_path)
    with open(experiment_path, "wb") as f:
        pickle.dump(stored_dict, f)


#import cProfile
#from torch.profiler import profile, record_function, ProfilerActivity
def load_and_sort_all_data(all_checkpoints, variances, dimensions, data_path, eval_dataset_size=4096,
                           data_extension=None, add_gauss_model=False, experiment='inflation', **eval_kwargs):
    """evaluate_space: fills out the whole embedding space to see which points get which likelihood assigned"""

    prog_per_dim = len(variances) * len(all_checkpoints)
    prog_per_sig = len(all_checkpoints)

    all_models = dict()
    total_eval_count = len(all_checkpoints) * len(dimensions) * len(variances)

    config = load_config(all_checkpoints['NICE'][dimensions[0]][variances[0]], data_path, verbose=False)
    store_path = create_stored_experiment_path(experiment, data_path, config, data_extension)
    print(f'Loading / Storing Data from\n{store_path}')

    for iDim, dimension in enumerate(dimensions):
        for iVar, var in enumerate(variances):
            #print(dimension, var)
            #if dimension != 36 or var != 1e-6:
            #    continue
            #if var == None or var == 0:
            #    continue

            # load config with all informations
            print('\r', f'dim {dimension} var {var} ', end='')
            config = load_config(all_checkpoints['NICE'][dimension][var], data_path, verbose=False)
            if data_extension == 'no rotation':
                assert config.no_rotation

            # create "true" likelihood as model
            setdefault_multiple_depth(all_models, 'true', dimension, var)

            # load experiment data if they already exist
            if was_experiment_loaded(experiment, data_path, config, data_extension):
                load_experiment_results(experiment, data_path, config, data_extension, all_models)
                continue

            true_model = _load_data_by_experiment(experiment, config, 2*eval_dataset_size if dimension == 2 else eval_dataset_size)

            all_models['true'][dimension][var] = true_model
            # for uniform latent distributions
            t_score_divide_by_var = False
            for iModel, model in enumerate(all_checkpoints.keys()):
                if 'RealVP' in model and dimension == 2:
                    continue  # coincides with NICE

                #print(model)
                progress = iDim * prog_per_dim + iVar * prog_per_sig + iModel + 1
                print('\r', f'Evaluate Model: {progress} / {total_eval_count} ', end='')

                setdefault_multiple_depth(all_models, model, dimension, var)
                current_model = all_models[model][dimension][var]

                torch_model = load_model_from_checkpoint(all_checkpoints[model][dimension][var], data_path)

                def _eval_model(data):
                    return evaluate_model(torch_model, data, var, t_score_divide_by_var=t_score_divide_by_var, **eval_kwargs)

                for data_key in true_model.keys():
                    current_model[data_key] = _eval_model(true_model[data_key][EMBEDDED_DATA])

                ##### state density sampling
                if experiment == 'inflation':
                    current_model['model_densities'], current_model['non_volume_model_densities'], current_model['model_normalizer'] = \
                        sample_model_densities(torch_model)

                del torch_model
                # correction term
                create_corrected_pds(current_model, true_model, var)

            if add_gauss_model:
                add_gauss_model(all_models, dimension, var, t_score_divide_by_var)

            # save s.t. it can be reloaded
            save_experiment_results(experiment, data_path, config, data_extension, all_models)

            # break
            torch.cuda.empty_cache()
        # break

    # RealNC Part
    if experiment in ['inflation', 'latent_transformation', 'full_space']:  # not interesting for RealNC
        # generate RealNC (RealNVP without volume change)
        all_models['RealNC'] = copy.deepcopy(all_models['RealNVP'])
        for dim in dimensions:
            for var in variances:
                # correction term
                nvp = all_models['RealNVP'][dim][var]

                real_nc_dict = all_models['RealNC'][dim][var]
                for data in nvp.keys():
                    if 'model_densities' in data or 'model_normalizer' in data: continue
                    nvp_data = nvp[data]
                    real_nc_dict[data] = {PD_IDX: nvp_data[NO_VOLUME_PD],
                                          T_SCORE: nvp_data[T_SCORE_NO_VOLUME],
                                          T_SCORE_NO_VOLUME: nvp_data[T_SCORE_NO_VOLUME],
                                          Z_IDX: nvp_data[Z_IDX]}

                if experiment == 'inflation':
                    real_nc_dict['model_densities'] = nvp['non_volume_model_densities']
                # corrected likelihoods
                create_corrected_pds(real_nc_dict, all_models['true'][dim][var], var)

                # RealNC needs the "normalizing constant" (NC) which is computed in corrected_pd
                # as "create_corrected_pds" needs the above schema we first create it and then
                # overwrite it. Note that CORRECTED_PD is not present for
                for data in nvp.keys():
                    if 'model_densities' in data or 'model_normalizer' in data: continue
                    if CORRECTED_PD in real_nc_dict[data].keys():
                        real_nc_dict[data]['pure lln'] = nvp[data][NO_VOLUME_PD]
                        real_nc_dict[data][PD_IDX] = real_nc_dict[data][CORRECTED_PD]

        # set all with var == 0 corrected
        normalize_zero_var_if_possible(all_models)

    return all_models


def remove_z_idx(all_models):
    for iDim, dimension in enumerate(all_models.keys()):
        for iVar, var in enumerate(all_models[dimension].keys()):
            for model in all_models[dimension][var].keys():
                for data_type in all_models[dimension][var][model].keys():
                    if 'model_densities' in data_type or 'model_normalizer' in data_type: continue
                    if Z_IDX in all_models[dimension][var][model][data_type].keys():
                        all_models[dimension][var][model][data_type].pop(Z_IDX)


def sample_model_densities(model):
    if isinstance(model, ResidualFlow):
        # dummy values
        return np.arange(10), np.arange(10), 10

    with torch.no_grad():
        #nice_implementation = model.implementation == 'nice' if 'implementation' in dir(model) else model.nice_implementation
        vp_model = model.implementation in ['nice', 'real_vp']

        sample_size = 10000# if vp_model else 4000  # VP models are easy to sample from
        x_log_prob, z_log_prob, log_det = model.sample_log_prob(sample_size)
        x_log_prob, z_log_prob, log_det = x_log_prob.cpu().numpy(), z_log_prob.cpu().numpy(), log_det.cpu().numpy()

        if vp_model:
            normalized_none_volume = x_log_prob  # is the same for NICE
            log_normalizer = None
        else:
            # starting from int 1/Z p_Z(f(x)) dx
            # Z = int p_Z(f(x)) q(x)/q(x) dx, q(x) = v(x) p_Z(f(x)) ...
            N = len(log_det)
            log_normalizer = -( -np.log(N) + logsumexp(-log_det) )
            normalized_none_volume = log_normalizer + z_log_prob

        return x_log_prob, normalized_none_volume, log_normalizer


def normalize_zero_var_if_possible(all_models):
    """if a normalization for sigma = 0 exists we make this standard.
    This is a better approximation """
    for model_name, model in all_models.items():
        if model_name == 'true':
            continue

        for dim in model.keys():
            if 0 in model[dim].keys():
                for data in model[dim][0].keys():
                    if 'model_densities' in data or 'model_normalizer' in data: continue
                    if CORRECTED_PD in model[dim][0][data].keys():
                        model[dim][0][data][PD_IDX] = model[dim][0][data][CORRECTED_PD]


def create_dataset_extension(folder):
    dataset_extension = ''
    for extension in ['no_rotation', 'no_scaling', 'prior-adjusted', '--new_loss', 'big_batch', 'adjusted_scaling',
                     'big_prior', 'trainable_scaling=0', 'trainable_scaling=1', 'big_dataset', 'go_down', '0.7', '0.9',
                     'conv', 'normal_init', 'samples=2000', 'samples=1000', 'uniform', 'tangent_noise_var=0.1']:
        if extension in folder:
            if dataset_extension != '':
                dataset_extension += '-'
            dataset_extension += extension

    return dataset_extension


def var_to_label(var):
    return 0 if var == 0 else f'${{10^ {{{int(np.log10(var))}}} }}$'

def gauss_symbol():
    return '$\mathcal{{N}}$'

def print_gauss_dist(var):
    return f'{gauss_symbol()}(0, {var_to_label(var)})'
