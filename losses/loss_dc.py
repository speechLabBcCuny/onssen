import torch
import torch.nn.functional as F
from utils import T, norm


def loss_dc(output, label):
    """
    adopted from nussl loss function:
    https://github.com/interactiveaudiolab/nussl/blob/master/nussl/transformers/transformer_deep_clustering.py
    inputs:
        output: a tuple containing a batch_size X T X F X embedding_dim tensor
        label: a tuple containing a batch_size X T X F X num_speaker tensor
    outputs:
        loss of deep clustering model/layer
    """
    assert len(output)==1, "Number of output must be 1 for Deep Clustering"
    assert len(label)==2, "Number of label must be 2 for Deep Clustering"
    embedding, = output
    label, mag_mix = label
    label = label.float()
    batch_size, frame_dim, frequency_dim, one_hot_dim = label.size()
    _, _, _, embedding_dim = embedding.size()

    embedding = embedding.view(batch_size, -1, embedding_dim)
    mag_mix = mag_mix.detach().view(batch_size, -1)
    label = label.view(batch_size, -1, one_hot_dim)

    # remove the loss of silence TF regions
    silence_mask = label.sum(2, keepdim=True)
    embedding = silence_mask * embedding

    # referred as weight WR
    # W_i = |x_i| / \sigma_j{|x_j|}
    weights = torch.sqrt(mag_mix / mag_mix.sum(1, keepdim=True))
    label = label * weights.view(batch_size, frame_dim*frequency_dim, 1)
    embedding = embedding * weights.view(batch_size, frame_dim*frequency_dim, 1)

    # do batch affinity matrix computation
    loss_est = norm(torch.bmm(T(embedding), embedding))
    loss_est_true = 2*norm(torch.bmm(T(embedding), label))
    loss_true = norm(torch.bmm(T(label), label))
    loss_embedding = loss_est - loss_est_true + loss_true

    return loss_embedding
