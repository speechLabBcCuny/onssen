"""
For every new loss function added, please import it here and add it to loss_fns
The loss function should take two arguments:
    output: a tuple from the network
    label: a tuple which is from dataloader
You need to assert the format of the output and label in the loss function!
"""
from .loss_chimera import loss_chimera_msa, loss_chimera_psa
from .loss_dc import loss_dc
from .loss_enhancement import loss_mask_msa, loss_mask_psa
from .loss_phase import loss_phase
from attrdict import AttrDict

def get_lossfns():
    loss_fns = AttrDict()
    loss_fns["loss_dc"] = loss_dc
    loss_fns["loss_chimera_msa"] = loss_chimera_msa
    loss_fns["loss_chimera_psa"] = loss_chimera_psa
    loss_fns["loss_phase"] = loss_phase
    loss_fns["loss_mask_msa"] = loss_mask_msa
    loss_fns["loss_mask_psa"] = loss_mask_psa
    return loss_fns
