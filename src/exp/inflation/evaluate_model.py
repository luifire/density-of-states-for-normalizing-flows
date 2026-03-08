from torch.utils.data import DataLoader
from scipy.special import logsumexp
from survae.experiments.image.utils import set_seeds

from precomputation.initialization import *
from misc.common_functions import load_config
from misc.constants import *


def evaluate_model(model, dataset, variance, t_score_divide_by_var,
                   ignore_deflation, cuda=True, batch_size=2048, normalize_realnc=True):
    """t_score_divide_by_var: true: is standard normalization, false is just '-mean'.
                                This needs to happen, as for uniform distribution '/ variance'
                                will erase intential spikeness of the learned models """
    result = dict()

    """config = load_config(checkpoint, data_path, verbose=False)
    model = load_model(checkpoint, verbose=False, cuda=cuda)
    model.eval()"""

    config = model.config

    if ignore_deflation:
        log_deflation = 0
    else:
        assert False, 'shouldnt happen'
        log_deflation = compute_log_deflation(variance, config.data_shape)

    bpd_factor = np.log(2) * np.prod(config.data_shape)

    with torch.no_grad():
        set_seeds(31019)

        mighty_prior = model.mighty_prior if get_bool_from_args('mighty_prior') else None

        # Loss Function, states=1 causes the loss function not apply the volume change caused by normalizing to [0,1]
        # (which for some reason was not directly encoded as a layer)
        loss_fn = Loss(config, model, config.data_shape, states=1, mighty_prior=mighty_prior)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        llh, Z, volumes = [], [], []
        for batch in loader:
            if cuda:
                batch = batch.cuda()
            lln, z, volume = loss_fn.x_to_full_info(batch)
            llh.append(lln), Z.append(z.cpu()), volumes.append(volume)

        llh, Z, volumes = torch.cat(llh), torch.cat(Z), torch.cat(volumes)
        result[Z_IDX], result[VOLUME_IDX] = Z, volumes

        llh = llh * bpd_factor

        #  no / bpd_factor as we don't do this for the true likelihood either
        result[PD_IDX] = llh - log_deflation

        # lln comes as bpd
        if config.flow == 'RealNVP':
            avg_volumes = compute_average_volume_change(model) if normalize_realnc else 0
            # this is not used anymore
            result[NO_VOLUME_PD] = llh - volumes + avg_volumes - log_deflation
            result[T_SCORE_NO_VOLUME] = result[NO_VOLUME_PD] - result[NO_VOLUME_PD].mean()

        # "standard normalize"
        result[T_SCORE] = result[PD_IDX] - result[PD_IDX].mean()

        if t_score_divide_by_var:
            raise Exception('Really?')
        #    result[T_SCORE] = result[T_SCORE] / result[LLH_IDX].var()
        #    result[T_SCORE_NO_VOLUME] = result[T_SCORE_NO_VOLUME] / result[NO_VOLUME_LLN].var()

        #result[CORRECTED_LLH], result[CORRECTION_TERM] = correct_using_true_distribution(result[T_SCORE], true_logprob)
        #result[CORRECTED_NO_VOL_LLH], result[CORRECTION_TERM_NO_VOL] = correct_using_true_distribution(result[T_SCORE_NO_VOLUME], true_logprob)

        # move to cpu and numpy
        result = {key: value.cpu().numpy() for key, value in result.items()}

    #del model

    return result


def compute_log_deflation(variance, data_shape):
    # See Theorem 5 in Horvat 2021 and footnote 5. q_n(x|x) = 1/(2πσ^2)^((D-d)/2) for 1D Gauss
    # => we assume knowledge about the dimensionality of the latent manifold
    # data_shape = (6 ,6 , 1)
    # times checked: II
    if variance == 0:
        return 0

    dim = np.prod(data_shape) - 1  # D-d
    return - dim/2 * (np.log(2) + np.log(np.pi) + np.log(variance))


def compute_average_volume_change(model):
    """This uses the learned distribution to evaluate the average volume change"""
    assert model.training is False

    used_samples = 1000
    batch_size = 1000
    volumes = []
    for _ in range(used_samples//batch_size):
        model.sample(batch_size)
        vol_change = -model.logdet()
        volumes.append(vol_change)

    volumes = torch.cat(volumes)
    volumes = volumes.cpu().numpy()
    """computes the average log volume"""
    N = len(volumes)
    # Vol in original space is logged
    avg_log_volume = logsumexp(-volumes) - np.log(N)
    return avg_log_volume
