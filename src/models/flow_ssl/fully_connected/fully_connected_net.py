import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyNeuralNetwork(nn.Module):

    def __init__(self, in_channels, latent_dim, out_channels):
        super(FullyNeuralNetwork, self).__init__()

        #self.fc_1 = nn.Linear(in_channels, out_channels)
        self.fc_1 = nn.Linear(in_channels, latent_dim)
        self.fc_2 = nn.Linear(latent_dim, latent_dim)
        self.fc_3 = nn.Linear(latent_dim, latent_dim)
        self.fc_4 = nn.Linear(latent_dim, out_channels)

        #print(f'In: {in_channels} Out: {out_channels}')

    def forward(self, x):
        original_shape = x.shape
        x = x.flatten(1, 3)
        #print(f'Input: {x.shape}')
        x = self.fc_1(x)

        x = F.relu(x)

        x = self.fc_2(x)
        x = F.relu(x)

        x = self.fc_3(x)
        x = F.relu(x)

        x = self.fc_4(x)
        x = x.view(len(x), 2*original_shape[1], original_shape[2], original_shape[3])
        return x
