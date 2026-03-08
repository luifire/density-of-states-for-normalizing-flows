import flow_ssl
import warnings

from master.utils.computational import create_normal_distribution
from master.utils.utils import get_bool_from_args
from master.data.data_master import simple_data_choice


def get_nf_model(args, data_shape, verbose=True):

    # Model
    if verbose:
        print('Building {} model...'.format(args.flow))
    model_cfg = getattr(flow_ssl, args.flow)

    if args.uniform_noise is None and args.var is None and args.dataset not in simple_data_choice:
        #pre_inflation_ilogits = get_bool_from_args('pre_inflation_ilogits')
        #args.ilogits = ilogits = not pre_inflation_ilogits
        warnings.warn('No iLogits!!!')
        args.ilogits = ilogits = False
    else:
        args.ilogits = ilogits = False

    skip_scaling = 'skip_scaling' in dir(args) and args.skip_scaling
    if args.flow == 'RealNVPTabular':
        assert False, 'This is not meant to be used (David)'
        #net = model_cfg(in_dim=feature_dim, hidden_dim=args.num_mid_channels, num_layers=args.num_blocks,
        #                num_coupling_layers=args.num_coupling_layers_per_scale, init_zeros=args.init_zeros)
    elif 'RealNVP' in args.flow:
        net = model_cfg(in_channels=data_shape[0], init_zeros=args.init_zeros, mid_channels=args.num_mid_channels,
                        num_blocks=args.num_blocks, num_scales=args.num_scales, st_type=args.st_type,
                        use_batch_norm=not args.no_batchnorm, img_shape=data_shape, skip=not args.no_skip,
                        latent_dim=args.latent_dim, ilogits=ilogits,
                        skip_scaling=skip_scaling)

    elif 'NICE' in args.flow:
        net = model_cfg(in_channels=data_shape[0], init_zeros=args.init_zeros, mid_channels=args.num_mid_channels,
                        num_blocks=args.num_blocks, num_scales=args.num_scales, st_type=args.st_type,
                        use_batch_norm=not args.no_batchnorm, img_shape=data_shape, skip=not args.no_skip,
                        latent_dim=args.latent_dim, implementation='nice', ilogits=ilogits,
                        skip_scaling=skip_scaling)
    elif 'RealVP' in args.flow:
        net = model_cfg(in_channels=data_shape[0], init_zeros=args.init_zeros, mid_channels=args.num_mid_channels,
                        num_blocks=args.num_blocks, num_scales=args.num_scales, st_type=args.st_type,
                        use_batch_norm=not args.no_batchnorm, img_shape=data_shape, skip=not args.no_skip,
                        latent_dim=args.latent_dim, implementation='real_vp', ilogits=ilogits,
                        skip_scaling=skip_scaling)
    elif args.flow == 'Glow':
        assert False, 'Should not be reached'
        net = model_cfg(image_shape=data_shape, mid_channels=args.num_mid_channels, num_scales=args.num_scales,
                        num_coupling_layers_per_scale=args.num_coupling_layers_per_scale, num_layers=args.num_blocks,
                        multi_scale=not args.no_multi_scale, st_type=args.st_type)
    if verbose:
        coupling_string = 'CouplingLayer'
        print(f'Model contains {str(net.body).count(coupling_string)} coupling layers')

    prior = create_normal_distribution(data_shape)
    net.set_prior(prior)

    return net