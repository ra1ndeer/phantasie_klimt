import torch
import torch.nn as nn


class KullbackLeiblerDivergence(nn.Module):
    """
    Implements the Kullback-Leibler Divergence of a
    distribution p relative to a distribution q, 
    D_{KL}[p || q], where p is a multivariate Gaussian
    distribution with diagonal covariance structure, and
    q is a standard multivariate Gaussian distribution, with
    the same dimensions as p.
    """
    def __init__(self):
        super(KullbackLeiblerDivergence, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(logvar + 1. - torch.exp(logvar) - mu**2)