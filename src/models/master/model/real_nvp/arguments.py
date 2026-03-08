import argparse

def add_nf_args(parser):

    # David:
    parser.add_argument('--new_loss', action='store_true', default=False, help='new loss created by David')
    parser.add_argument('--new_loss_2', action='store_true', default=False, help='new loss created by David')
    parser.add_argument('--loss_sample_count', type=int, default=16, help='new loss created by David')
    parser.add_argument('--new_loss_exp', default=1.0, type=float, help='the scale of the harmonic mean')

    parser.add_argument('--no_scaling_layer', action='store_true', default=False,
                        help='For NICE: will not add the scaling layer')

    # NF
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--use_validation', action='store_true', help='Use trainable validation set')

    parser.add_argument('--flow', type=str, default="RealNVP", help='Flow model to use (default: RealNVP)')

    parser.add_argument('--num_blocks', default=8, type=int, help='number of blocks in ResNet')
    parser.add_argument('--num_scales', default=2, type=int, help='number of scales in multi-layer architecture')

    parser.add_argument('--num_mid_channels', default=64, type=int, help='number of channels in coupling layer parametrizing network')
    parser.add_argument('--st_type', choices=['highway', 'resnet', 'convnet', 'autoencoder_old', 'autoencoder', 'resnet_ae', 'fully'],
                        default='fully')
    parser.add_argument('--latent_dim', default=100, type=int, help='dim of bottleneck in autoencoder st-network')
    parser.add_argument('--no_batchnorm', action='store_true')
    parser.add_argument('--no_skip', action='store_true')

    parser.add_argument('--init_zeros', action='store_true')
    parser.add_argument('--optim', choices=['Adam', 'RMSprop'], default='Adam')

    # Flow-VAE params
    parser.add_argument("--decoder_likelihood", type=str, default="gaussian",
        choices=["gaussian", "binary_ce"], help="Decoder likelihood",)
    parser.add_argument("--logvar_num_hidden_layers", type=int, default=1, help="Number of hidden layers for logvar")
    parser.add_argument("--logvar_num_hidden_units", type=int, default=500, help="Number of hidden units for logvar")
    parser.add_argument('--reconstruction_loss', action='store_true')
    parser.add_argument('--reconstruction_weight', default=1., type=float, help='weight of the reconstruction loss term')
    parser.add_argument('--reconstruction_rampup', default=1, type=int, help='Number of epochs for reconstruction loss rampup')