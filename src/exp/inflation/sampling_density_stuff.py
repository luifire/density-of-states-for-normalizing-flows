from precomputation.initialization import *
from inflation.jupyter_helper import *
from master.training.utils import initialize_model


def compute_sample_densities(checkpoint, data_path, sample_count):
    """Samples from the model and returns the sampled likelihood and the prior likelihood"""

    samples_per_batch = 1024

    config = load_config(checkpoint, data_path, verbose=False)
    model = load_model(checkpoint, verbose=False)
    initialize_model(model, config)

    loss_fn = Loss(config, model, config.data_shape, states=1)
    model.eval()

    sample_stack, prior_stack = [], []
    with torch.no_grad():
            for i in range(sample_count // samples_per_batch):
                # s.t. plotting looks better
                samples, sampled_z, x_log_prob, z_log_prob = model.sample_with_log_prob(samples_per_batch,
                                                                                        random_state=i)
                sample_stack.append(x_log_prob.cpu().numpy())
                prior_stack.append(z_log_prob.cpu().numpy())

                """
                this makes some double checks
                
                lln, computed_z, volume = loss_fn.x_to_full_info(samples)

                if config.flow == 'NICE':
                    if not (volume == volume[0]).all():
                        raise Exception('Nice changed volume!')

                if not torch.allclose(computed_z, sampled_z, atol=1e-3):
                    raise Exception('computed Z and sampled Z not equal!')

                if not torch.allclose(lln * (np.log(2) * np.prod(config.data_shape)), x_log_prob, atol=1e-5):
                    raise Exception('LLN do not align!')
                """

    samples = np.stack(sample_stack).flatten()
    prior_lp = np.stack(prior_stack).flatten()
    del model

    return samples, prior_lp
