
def print_relevant_argument_info(args, verbose=True):

    super_relevant = ['wandb_name_ext', 'debug']
    relevant = ['epochs', 'dataset', 'batch_size', 'augmentation', 'lr', 'weight_decay']

    nf_specific = ['flow', 'prior', 'num_blocks', 'num_scales', 'num_coupling_layers_per_scale', 'num_mid_channels',
                   'st_type', 'latent_dim', 'optim', 'new_loss']
    survae_specific = ['num_scales', 'num_steps', 'actnorm', 'pooling', 'dequant', 'optimizer', 'momentum', 'gamma',
                       'new_loss']
    iwae_specific = ['num_k', 'latent_dim', 'kld_weight']

    def _print_keys(keys):
        for key in keys:
            if key in dir(args) and args.__getattribute__(key) != '':
                print(f'{key}: {args.__getattribute__(key)}')

    print_line()
    print(model_short_to_name(args.model))

    if verbose == 1:
        _print_keys(super_relevant)
        return

    _print_keys(relevant)
    if args.model == 'survae':
        _print_keys(survae_specific)
        repo = 'https://github.com/didriknielsen/survae_flows   (there is no good explanation...)'
    elif args.model == 'nf':
        _print_keys(nf_specific)
        repo = 'https://github.com/PolinaKirichenko/flows_ood' \
               '/blob/e9755db3454bbf9e8d46086446ddb9caa8870173/experiments/train_flows/train_unsup.py'
    elif args.model == 'iwae':
        _print_keys(iwae_specific)
        repo = 'https://github.com/AntixK/PyTorch-VAE'
    else:
        raise NotImplementedError()

    print('\nFor information about certain arguments see: ')
    print(repo)

    print_line()


def model_short_to_name(name):
    if isinstance(name, str) is False:
        return name
    name = name.lower()
    name_translator = {'nf': 'Normalizing Flow',
                       'nf_new_loss': 'Normalizing Flow - new loss',
                       'nf_no_bias': 'Normalizing Flow - no bias',
                       'nice': 'NICE',
                       'continued_new_loss': 'pretrained NF, continued with new loss',
                       'survae': 'SurVAE',
                       'survae_no_bias': 'SurVAE - no bias',
                       'survae_new_loss': 'SurVAE - new loss',
                       'survae_hierarshow_loss_histogramschy': 'SurVAE - pyramid shape',
                       'iwae': 'Importance Weighted VAE',
                       'iwae_no_bias': 'Importance Weighted VAE - no bias',
                       'iwae_with_bias': 'Importance Weighted VAE',
                       'expert': 'Mixture of Experts',
                       }
    if name in name_translator:
        return name_translator[name]
    return name


def print_line():
    print('----------------------------------------------------------------------------------------------------------')
