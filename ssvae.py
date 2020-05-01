"""
    Module for Style2Vec

    @author: Younghyun Kim
    @edited: 2020.04.03.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy import prod, sqrt
from distributions.sparse import Sparse
from torch.distributions import Laplace, Normal

class StyleScoreVAE(nn.Module):
    """
        Style Score VAE Class
    """
    def __init__(self, input_dim,
                 ss_dim=16, hidden_dim=32,
                 prior_dist=Sparse,
                 posterior_dist=Normal,
                 likelihood_dist=Laplace,
                 learn_prior_variance=False,
                 prior_variance_scale=1.,
                 pi=0.8):
        super(StyleScoreVAE, self).__init__()
        self.input_dim = input_dim
        self.ss_dim = ss_dim
        self.hidden_dim = hidden_dim
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = Enc(input_dim, ss_dim, hidden_dim)
        self.dec = Dec(input_dim, ss_dim, hidden_dim)
        self.learn_prior_variance = learn_prior_variance
        self.prior_variance_scale = prior_variance_scale
        self.modelName = 'StyleScoreVAE'

        self._pz_mu, self._pz_logvar = self.init_pz()
        self.pi = nn.Parameter(torch.tensor(pi), requires_grad=False)

    def get_style_score(self, data, K=1):
        " Get Style Score Embeddings "
        qz_x = self.qz_x(*self.enc(data))
        style_score = qz_x.rsample(torch.Size([K]))

        if K > 1:
            style_score = style_score.mean(0)
        else:
            style_score = style_score.squeeze(0)

        return style_score

    def generate(self, N, K):
        " generation "
        self.eval()
        with torch.no_grad():
            mean_pz = self.pz_params[1]
            mean = self.dec(mean_pz)[0]
            pz = self.pz(*self.pz_params)

            px_z_params = self.dec(pz.sample(torch.Size([N])))
            means = px_z_params[0]
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, means, samples

    def reconstruct(self, data):
        " reconstruction "
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample())

        return px_z_params[0]

    def forward(self, x, K=1, no_dec=False):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        if K > 1:
            zs = zs.mean(0)
        else:
            zs = zs.squeeze(0)
        if no_dec:
            return qz_x, zs
        px_z = self.px_z(*self.dec(zs))

        return qz_x, px_z, zs

    @property
    def device(self):
        " show device "
        return self._pz_mu.device

    @property
    def pz_params(self):
        return self.pi.mul(1), self._pz_mu.mul(1), \
                torch.sqrt(self.prior_variance_scale * \
                self._pz_logvar.size(-1) * F.softmax(self._pz_logvar, dim=1))

    def init_pz(self):
        " pz initialization "
        # means
        pz_mu = nn.Parameter(torch.zeros(1, self.ss_dim),
                             requires_grad=False)

        # variances
        logvar = torch.zeros(1, self.ss_dim)
        pz_logvar = nn.Parameter(logvar,
                                 requires_grad=self.learn_prior_variance)

        return pz_mu, pz_logvar

class Enc(nn.Module):
    " Encoder class "
    def __init__(self, input_dim, ss_dim=16, hidden_dim=32):
        super(Enc, self).__init__()

        self.input_dim = input_dim
        self.ss_dim = ss_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, ss_dim)
        self.fc_sig = nn.Linear(hidden_dim, ss_dim)

    def forward(self, x):
        hidden = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        return self.fc_mu(hidden), F.softplus(self.fc_sig(hidden))

class Dec(nn.Module):
    " Decoder class "
    def __init__(self, output_dim, ss_dim=16, hidden_dim=32):
        super(Dec, self).__init__()

        self.output_dim = output_dim
        self.ss_dim = ss_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(ss_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, c=0.1):
        hidden = F.leaky_relu(self.fc1(z), 0.2, inplace=True)
        return F.relu(self.fc2(hidden), inplace=True), \
                torch.tensor(c).to(z.device)  # or 0.05
