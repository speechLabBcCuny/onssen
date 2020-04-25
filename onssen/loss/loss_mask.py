from .loss_util import norm, norm_1d
import torch
import torch.nn.functional as F


def loss_mask_msa(output, label):
    """
    The loss function of Magnitude Spectrum Approximation (MSA).
    It is for enhancing speech in a noisy recording.
    output:
        mask: batch_size X T X F  tensor
    label:
        mag_noisy: the magnitude of mix speech
        mag_clean: the magnitude of clean speech s1
    """
    [clean_est] = output
    [mag_clean, cos_diff] = label
    #compute the loss of mask part
    # loss = nn.MSELoss()(mask * mag_noisy, mag_clean)

    loss = torch.nn.MSELoss()(clean_est, mag_clean)
    return loss


def loss_mask_psa(output, label):
    """
    The loss function of Phase-sensitive Spectrum Approximation (PSA).
    It is for enhancing speech in a noisy recording.
    output:
        mask: batch_size X T X F  tensor
    label:
        mag_noisy: the magnitude of mix speech
        mag_clean: the magnitude of clean speech s1
        cos_diff: the cosine of phase difference between mix and clean
    """
    [mask] = output
    [mag_noisy, mag_clean, cos_diff] = label
    #compute the loss of mask part
    loss = norm_1d(mask * mag_noisy - torch.min(mag_noisy,F.relu(mag_clean*cos_diff)))
    return loss
