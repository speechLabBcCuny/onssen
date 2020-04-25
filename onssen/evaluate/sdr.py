### Forked from https://github.com/yluo42/TAC/blob/master/utility/sdr.py
import numpy as np
from itertools import permutations
from torch.autograd import Variable

import scipy,time,numpy

import torch

# Pytorch implementation with batch processing
def calc_sdr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: an optional mask for sequence masking. This is for cases where zero-padding was applied at the end and should not be consider for SDR calculation.
    """

    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask

    def calculate(estimation, origin):
        origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)
        scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)

        est_true = scale * origin  # (batch, nsample)
        est_res = estimation - est_true  # (batch, nsample)

        true_power = torch.pow(est_true, 2).sum(1) + 1e-8
        res_power = torch.pow(est_res, 2).sum(1) + 1e-8

        return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, )

    best_sdr = calculate(estimation, origin)

    return best_sdr


def batch_SDR_torch(estimation, origin, mask=None, return_perm=False):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    mask: optional, (batch, nsample), binary
    return_perm: bool, whether to return the permutation index. Default is false.
    """

    batch_size_est, nsource_est, nsample_est = estimation.size()
    batch_size_ori, nsource_ori, nsample_ori = origin.size()

    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."

    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."

    batch_size = batch_size_est
    nsource = nsource_est

    # zero mean signals
    estimation = estimation - torch.mean(estimation, 2, keepdim=True).expand_as(estimation)
    origin = origin - torch.mean(origin, 2, keepdim=True).expand_as(estimation)

    # SDR for each permutation
    SDR = torch.zeros((batch_size, nsource, nsource)).type(estimation.type())
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr_torch(estimation[:,i], origin[:,j], mask)

    # choose the best permutation
    SDR_max = []
    SDR_perm = []
    perm = sorted(list(set(permutations(np.arange(nsource)))))
    for permute in perm:
        sdr = []
        for idx in range(len(permute)):
            sdr.append(SDR[:,idx,permute[idx]].view(batch_size,-1))
        sdr = torch.sum(torch.cat(sdr, 1), 1)
        SDR_perm.append(sdr.view(batch_size, 1))
    SDR_perm = torch.cat(SDR_perm, 1)
    SDR_max, SDR_idx = torch.max(SDR_perm, dim=1)

    if not return_perm:
        return SDR_max / nsource
    else:
        return SDR_max / nsource, SDR_idx
