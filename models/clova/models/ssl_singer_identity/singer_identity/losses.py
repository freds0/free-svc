import torch
import torch.nn.functional as F
from singer_identity.utils.core import similarity, roll


def std_batch(x, var=1, eps=1e-8):
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(var - std))


def variance_hinge_reg(x, y, var=1):
    # From https://github.com/facebookresearch/vicreg
    std_x = std_batch(x, var=var)
    std_y = std_batch(y, var=var)
    std_loss = std_x / 2 + std_y / 2
    return std_loss


def covariance(x):
    # In official implementation they do mean over batch (to verify)
    # mean = x.mean(1, keepdims=True)
    mean = x.mean(dim=0)
    x = x - mean
    cov = torch.matmul(x.transpose(0, 1), x) / (x.shape[0] - 1)
    # cov = (x.T @ x) / (x.shape[0] - 1)
    return cov


def covariance_reg(x, y):
    eye = torch.eye(x.shape[1]).to(x.device)
    cov_x = covariance(x)
    cov_y = covariance(y)
    assert cov_x.shape[0] == cov_x.shape[1]
    assert cov_y.shape[0] == cov_y.shape[1]
    cov_reg = (cov_x * (1 - eye)).pow(2).sum() / x.shape[1] + (cov_y * (1 - eye)).pow(
        2
    ).sum() / x.shape[1]
    return cov_reg


def invariance_loss(x, y):
    return F.mse_loss(x, y)


def vicreg_loss(x, y, gamma=1, fact_inv_loss=1, fact_var=1, fact_cov=1):
    # Adapted from https://github.com/facebookresearch/vicreg
    repr_loss = invariance_loss(x, y)
    std_loss = variance_hinge_reg(x, y, var=gamma)
    cov_loss = covariance_reg(x, y)
    loss = fact_inv_loss * repr_loss + fact_var * std_loss + fact_cov * cov_loss
    return loss


def compute_norms(*args):
    norms = []
    for arg in args:
        norms.append(torch.sqrt((arg**2).sum(1)))
    return norms


def align_loss(x, y, alpha=2):
    # From https://github.com/SsnL/align_uniform
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    # From https://github.com/SsnL/align_uniform
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def contrastive_loss(z1, z2, temp=0.2, nr_negative=1, decouple=False):
    cost_pos = similarity(z1, z2, temp)  # Positive samples
    cost_neg = []

    n_rolls = min(z1.shape[0] - 1, nr_negative)  # Number of negative samples
    curr_neg_z = z2

    for i in range(n_rolls):
        curr_neg_z = roll(curr_neg_z)  # Shifts batch
        cost_neg.append(similarity(z1, curr_neg_z, temp))  # Negative sim.

    if not decouple:
        cost_neg.append(cost_pos)  # Adds positive similarity in denominator

    cost_neg = torch.stack(cost_neg).transpose(1, 0)
    cost = (-cost_pos + torch.logsumexp(cost_neg, 1)).mean()
    # TODO: implement similarities with less operations, but this works
    ratio = torch.mean(cost_neg) / (
        torch.mean(cost_pos) + torch.tensor(1e-6).type_as(z1)
    )
    return cost, ratio.item()
