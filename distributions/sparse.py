"""
    Module for Sparse Distribution

    @author: Younghyun Kim
    @edited: 2020.04.03.
"""

import torch
import torch.distributions as dist
from torch.distributions.utils import broadcast_all
from numbers import Number

class Sparse(dist.Distribution):
    has_rsample = False

    def __init__(self, pi, loc, scale, alpha=0.05, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.pi = pi
        self.alpha = torch.tensor(alpha).to(self.loc.device)
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super(Sparse, self).__init__(batch_shape)

    @property
    def mean(self):
        return (1 - self.pi) * self.loc

    @property
    def stddev(self):
        return self.pi * self.alpha + (1 - self.pi) * self.scale

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.bernoulli(self.pi * torch.ones(shape).to(self.loc.device))
        res = p * self.alpha * torch.randn(shape).to(self.loc.device) + \
                (1 - p) * (self.loc + self.scale * \
                torch.randn(shape).to(self.loc.device))
        return res

    def log_prob(self, value):
        res = torch.cat([(dist.Normal(torch.zeros_like(self.loc),
            self.alpha).log_prob(value) + self.pi.log()).unsqueeze(0),
            (dist.Normal(self.loc, self.scale).log_prob(value) + \
                    (1 - self.pi).log()).unsqueeze(0)], dim=0)
        return torch.logsumexp(res, 0)

