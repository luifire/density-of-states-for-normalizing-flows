import torch.nn as nn
import torch
from torch.nn.functional import softmax
from torch.distributions import Normal, StudentT, Cauchy, Gumbel, Laplace, Uniform, Multinomial
from scipy.special import gamma
import numpy as np


class MightyPrior(nn.Module):
    def __init__(self, args):
        super(MightyPrior, self).__init__()
        self.args = args

        # note: I learn these parameters only for one single Generalized Normal even though there are many
        self.gen_normal_log_beta = nn.Parameter((3 * MightyPrior._ones(1)).log())
        self.gen_normal = Generalized_Normal(MightyPrior._zeros(1), (3 * MightyPrior._ones(1)).log())
        # RealNVP produces Outputs of 200 at first run => gradient explosion due to exp
        self.gumbel_scale = nn.Parameter(100 * MightyPrior._ones(1))
        self.gumbel = MightyGumbel(self.gumbel_scale)


        self.prior_list = [Normal(MightyPrior._zeros(1), MightyPrior._ones(1)),
                           #StudentT(df=MightyPrior._ones(1)),
                           #Gumbel(loc=MightyPrior._zeros(1), scale=3 * MightyPrior._ones(1)),
                           self.gumbel,
                           Laplace(loc=MightyPrior._zeros(1), scale=MightyPrior._ones(1)),
                           #Generalized_Normal(MightyPrior._zeros(1), (1 * MightyPrior._ones(1)).log()),
                           #Generalized_Normal(MightyPrior._zeros(1), (2 * MightyPrior._ones(1)).log()),
                           #Generalized_Normal(MightyPrior._zeros(1), (8 * MightyPrior._ones(1)).log())]
                           self.gen_normal]

                           #Cauchy(loc=MightyPrior._zeros(1), scale=MightyPrior._ones(1)),  # tendenziell eher raus - weil entartet
                           #SmoothUniform()]

        print('!!!!Prior List!!!!!!')
        print(self.prior_list)
        self.img_size = np.prod(args.data_shape)
        weights = torch.nn.init.normal_(torch.zeros(self.img_size, len(self.prior_list)), mean=0.0, std=0.01).cuda()
        self.weights = nn.Parameter(weights)
        self.eval_flip = False

    def forward(self, z, sldj, mean):
        """final_log_prob = self.prior_list[0].log_prob(z).sum(1)
        final_log_prob = final_log_prob + sldj
        # bpd
        bpd = final_log_prob / (np.log(2) * z.shape[1])
        # create loss
        loss = -bpd.mean() if mean else -bpd
        return loss"""
        #self._print_weights()
        if torch.isnan(z).any() or torch.isinf(z).any():
            assert False, z

        #if z.abs().max() > 100:
        #    print(z.abs().max())

        prior_log_probs = []
        for prior in self.prior_list:
            log_prob = prior.log_prob(z)
            prior_log_probs.append(log_prob)

        prior_log_probs = torch.stack(prior_log_probs)
        # batch, variables, prior
        prior_log_probs = prior_log_probs.transpose(0, 1).transpose(1, 2)

        weights = softmax(self.weights, dim=1)

        """one latent x: p(x) = sum w_i p_i(x) | log sum w_i exp(log p_i(x)) = log sum exp( log p_i(x) + log(w_i))
        could of course be computed in none-log space but is probably more stable this way"""
        prior_probs = prior_log_probs.exp()
        weighted_prior_probs = prior_probs * weights
        # summing in none-log space / sum w_i p_i
        summed_prior_results = weighted_prior_probs.sum(2)
        final_log_prob = summed_prior_results.log().sum(1) + sldj
        #weighted_log_probs = prior_results + weights.log()
        #weighted_log_prob_sum = torch.logsumexp(weighted_log_probs, dim=2)
        #final_log_prob = weighted_log_prob_sum.sum(1) + sldj

        # bpd
        bpd = final_log_prob / (np.log(2) * z.shape[1])
        # create loss
        loss = -bpd.mean() if mean else -bpd
        return loss

    def sample(self, num):

        samples_of_all_priors = []
        for prior in self.prior_list:
            z_sampled = prior.sample((num, self.img_size)).squeeze(2)
            samples_of_all_priors.append(z_sampled)

        samples_of_all_priors = torch.stack(samples_of_all_priors)
        samples_of_all_priors = samples_of_all_priors.transpose(0, 1).transpose(1, 2)

        weights = softmax(self.weights, dim=1)
        multinomial = Multinomial(probs=weights)
        picks = multinomial.sample((num, ))
        # controled: I
        z = samples_of_all_priors[picks == 1].view((num, self.img_size))
        return z

    def print_weights(self):
        # flipflop to print weights
        #if self.training is False and self.eval_flip is False:
        #    self.eval_flip = True
        print('mighty prior weights')
        #self.gen_normal.print_params()
        self.gumbel.print_params()

        weights = softmax(self.weights, dim=1).detach().cpu()
        argmax = weights.argmax(dim=1)
        print(f'Used Prior Bin Count \n{torch.bincount(argmax)}')
        #argmax = weights.argmax(dim=1, keepdim=True)
        #print(f'Weights for these priors \n{torch.gather(weights, 1, argmax).squeeze(1)}')

        #if self.training and self.eval_flip:
        #    self.eval_flip = False

    @staticmethod
    def _ones(k):
        return torch.ones(k).cuda()

    @staticmethod
    def _zeros(k):
        return torch.zeros(k).cuda()


class SmoothUniform():

    def __init__(self, variance=1e-2):
        self.var = variance * torch.ones(1).cuda()
        self.uniform = Uniform(MightyPrior._zeros(1), MightyPrior._ones(1))
        self.normal = Normal(MightyPrior._zeros(1), np.sqrt(variance) * MightyPrior._ones(1))

    def log_prob(self, x):
        return 1/2 * (torch.erf((0.5 - x) / torch.sqrt(self.var * 2)) -
                      torch.erf((-0.5 - x) / torch.sqrt(self.var * 2)))

    def sample(self, shape):
        uni_sample = self.uniform.sample(shape)
        normal_sample = self.normal.sample(shape)
        return uni_sample + normal_sample

class MightyGumbel():

    def __init__(self, scale):
        self.scale = scale

    def log_prob(self, x):
        """Copied from Torch """
        y = x / self.scale
        return (y - y.exp()) - self.scale.log()

    def sample(self, sample_shape):
        gumbel_sampler = Gumbel(loc=MightyPrior._zeros(1), scale=self.scale)

        return gumbel_sampler.sample(sample_shape)

    def print_params(self):
        beta = self.scale.detach().cpu()
        print(f'Gumbel beta: {beta}')


class Generalized_Normal():

    def __init__(self, log_alpha, log_beta):
        self.log_alpha = log_alpha
        self.log_beta = log_beta

    def log_prob(self, x):
        # to make it positive
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()

        return self.log_beta - np.log(2) - self.log_alpha - torch.lgamma(1 / beta) - \
               (torch.abs(x) / alpha) ** beta

    def sample(self, sample_shape):
        #https: // cran.r - project.org / web / packages / gnorm / vignettes / gnormUse.html
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()

        shape = 1 + 1 / beta
        rate = 2**(- beta)

        gamma = torch.distributions.Gamma(concentration=shape, rate=rate)
        gamma_samples = gamma.sample(sample_shape)

        delta = alpha * gamma_samples ** (1 / beta) / np.sqrt(2)

        uniform = Uniform(low=-delta, high=delta)
        samples = uniform.sample((1, )).view(sample_shape).unsqueeze(2)
        return samples

    def print_params(self):
        beta = self.log_beta.exp().detach().cpu()

        print(f'Generalized Normal beta: {beta}')