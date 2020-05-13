"""
For every new loss function added, please import it here and add it to loss_fns
The loss function should take two arguments:
    output: a tuple from the network
    label: a tuple which is from dataloader
You need to assert the format of the output and label in the loss function!
"""
from .loss_dc import loss_dc
from .loss_chimera import loss_chimera_msa, loss_chimera_psa
from .loss_mask import loss_mask_msa, loss_mask_psa
from .loss_phase import loss_phase
from .loss_e2e import SI_SNR, permute_SI_SNR, sisnr, si_snr_loss
from .loss_util import T, norm, norm_1d


__all__ = [
    'loss_dc', 'loss_chimera_msa', 'loss_chimera_psa',
    'loss_mask_msa', 'loss_mask_psa',
    'loss_phase',
    'SI_SNR', 'permute_SI_SNR', 'sisnr', 'si_snr_loss',
]
