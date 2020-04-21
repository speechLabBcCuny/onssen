from utils import T, norm, norm_1d
from .loss_dc import loss_dc
import torch
import torch.nn.functional as F

def loss_phase(output, label):
    assert len(output) == 6, "There must be 5 tensors in the output"
    assert len(label) == 6, "There must be 6 tensors in the label"
    [embedding, mask_A, mask_B, phase_A, phase_B] = output
    [one_hot_label, mag_mix, mag_s1, mag_s2, phase_s1, phase_s2] = label
    batch_size, time_size, frequency_size = mag_mix.size()
    # compute the loss of embedding part
    loss_embedding = loss_dc([embedding, mag_mix], [one_hot_label])

    #compute the loss of mask part
    loss_mask1 = norm_1d(mask_A*mag_mix - mag_s1)\
               + norm_1d(mask_B*mag_mix - mag_s2)
    loss_mask2 = norm_1d(mask_B*mag_mix - mag_s1)\
               + norm_1d(mask_A*mag_mix - mag_s2)

    amin = loss_mask1<loss_mask2
    loss_mask = torch.zeros_like(loss_mask1)
    loss_mask[amin] = loss_mask1[amin]
    loss_mask[~amin] = loss_mask2[~amin]

    loss_phase1 = -mag_mix * F.cosine_similarity(phase_A, phase_s1, dim=3)\
                  -mag_mix * F.cosine_similarity(phase_B, phase_s2, dim=3)
    loss_phase2 = -mag_mix * F.cosine_similarity(phase_B, phase_s1, dim=3)\
                  -mag_mix * F.cosine_similarity(phase_A, phase_s2, dim=3)

    loss_phase1 = torch.sum(loss_phase1.reshape(batch_size,-1),dim=1)
    loss_phase2 = torch.sum(loss_phase2.reshape(batch_size,-1),dim=1)
    loss_phase = torch.zeros_like(loss_phase1)
    loss_phase[amin] = loss_phase1[amin]
    loss_phase[~amin] = loss_phase2[~amin]

    return loss_embedding*0.975 + loss_mask*0.025 + loss_phase*0.025
