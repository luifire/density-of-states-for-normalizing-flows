import torch.optim as optim
from demun.demun import Demun

def get_demun_model(args, data_shape):
    return Demun(target_shape=data_shape,
                latent_dim=args.latent_dim,
                num_samples=args.num_k)

def get_demun_loss(args, mean):
    """Call imitated from:
    https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/experiment.py"""
    #m_n = args.batch_size / len(dataset)
    def loss(demun, batch):
        results = demun.create_distributions()
        train_loss = demun.loss_function(results, batch, per_sample=not mean)['loss']
        return train_loss

    return loss


def get_demun_optimizer(args, model):
    # David: eps=1e-4 proposed by IWAE
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
