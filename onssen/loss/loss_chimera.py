from .loss_util import T, norm, norm_1d
from .loss_dc import loss_dc
import torch
import torch.nn.functional as F

def loss_chimera_msa(output, label):
    """
    output:
        noisy_mag: batch_size X T X F tensor
        masks: batch_size X T X F X num_speaker tensor
        clean_mags: batch_size X T X F X num_speaker tensor
    label:
        one_hot_label: the label for deep clustering
        mag_mix: the magnitude of mix speech
        mag_s1: the magnitude of clean speech s1
        mag_s2: the magnitude of clean speech s2
    """
    [embedding, mask_A, mask_B] = output
    [one_hot_label, mag_mix, mag_s1, mag_s2] = label
    batch_size, frame, frequency = mask_A.shape
    # compute the loss of embedding part
    loss_embedding = loss_dc([embedding], [one_hot_label, mag_mix])

    #compute the loss of mask part
    loss_mask1 = norm_1d(mask_A*mag_mix - mag_s1)\
               + norm_1d(mask_B*mag_mix - mag_s2)
    loss_mask2 = norm_1d(mask_B*mag_mix - mag_s1)\
               + norm_1d(mask_A*mag_mix - mag_s2)
    loss_mask = torch.min(loss_mask1, loss_mask2)

    return loss_embedding*0.975 + loss_mask*0.025

def loss_chimera_psa(output, label):
    """
    output:
        noisy_mag: batch_size X T X F tensor
        masks: batch_size X T X F X num_speaker tensor
        clean_mags: batch_size X T X F X num_speaker tensor
    label:
        one_hot_label: the label for deep clustering
        mag_mix: the magnitude of mix speech
        mag_s1: the magnitude of clean speech s1
        mag_s2: the magnitude of clean speech s2
        cos_s1: the cosine of phase difference between mix and s1
        cos_s2: the cosine of phase difference between mix and s2
    """
    [embedding, mask_A, mask_B] = output
    [one_hot_label, mag_mix, mag_s1, mag_s2, cos_s1, cos_s2] = label
    batch_size, frame, frequency = mask_A.shape
    # compute the loss of embedding part
    loss_embedding = loss_dc([embedding], [one_hot_label, mag_mix])
    #compute the loss of mask part
    loss_mask1 = norm_1d(mask_A*mag_mix - torch.min(mag_mix,F.relu(mag_s1*cos_s1)))\
               + norm_1d(mask_B*mag_mix - torch.min(mag_mix,F.relu(mag_s2*cos_s2)))
    loss_mask2 = norm_1d(mask_B*mag_mix - torch.min(mag_mix,F.relu(mag_s1*cos_s1)))\
               + norm_1d(mask_A*mag_mix - torch.min(mag_mix,F.relu(mag_s2*cos_s2)))
    loss_mask = torch.min(loss_mask1, loss_mask2)

    return loss_embedding*0.975 + loss_mask*0.025
