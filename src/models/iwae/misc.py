import torch.optim as optim

from iwae.iwae import IWAE


def add_iwae_args(parser):
    """Arguments for the iwae. 16 samples from the WAIC paper. """
    parser.add_argument("--num_k", type=int, default=16, help="num of samples used in importance weighted ELBO")
    parser.add_argument("--latent_dim", type=int, default=128, help="size of the latent space")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--kld_weight", type=float, default=0.001, help="the weight for the KL-Term. 'Commonly used "
                                                                        "to avoid posterior collapse' (WAIC)")
    parser.add_argument("--depth", type=int, default=5, choices=[3, 4, 5, 6], help="how many layers do we want. "
                                                                                      "Size increases exponentially (2^(i+4))")


def get_iwae_model(args, data_shape):

    return IWAE(data_shape=data_shape,
                latent_dim=args.latent_dim,
                num_samples=args.num_k,
                kld_train_weight=args.kld_weight,
                depth=args.depth)


def get_iwae_loss(args, mean):
    """Call imitated from:
    https://github.com/AntixK/PyTorch-VAE/blob/8700d245a9735640dda458db4cf40708caf2e77f/experiment.py"""
    #m_n = args.batch_size / len(dataset)
    def loss(iwae, batch):
        results = iwae.forward(batch)
        train_loss = iwae.loss_function(*results, per_sample=not mean)['loss']
        return train_loss

    return loss


def get_iwae_optimizer(args, model):
    # David: eps=1e-4 proposed by IWAE
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
