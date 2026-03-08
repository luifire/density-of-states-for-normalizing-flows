import torch.optim as optim
from master.model.real_nvp.utils import get_param_groups

# this is all copied from flows_ood
def get_nf_optimizer(args, model):

    if args.flow in ['RealNVP', 'NICE', 'RealVP'] and args.flow != 'RealNVPTabular':
        # We need this to make sure that weight decay is only applied to g -- norm parameter in Weight Normalization
        param_groups = get_param_groups(model, args.weight_decay, norm_suffix='weight_g')
        if args.optim == 'Adam':
            optimizer = optim.Adam(param_groups, lr=args.lr)
        else:
            optimizer = optim.RMSprop(param_groups, lr=args.lr)

    elif args.flow == 'Glow' or args.flow == 'RealNVPTabular':
        if args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer