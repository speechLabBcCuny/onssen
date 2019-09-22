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
    assert len(label)==1, "Number of label must be 1 for Deep Clustering"
    embedding = output[0]
    label = label[0].float()
    batch_size, time_size, frequency_size,one_hot_dim = label.size()
    _, _, _, embedding_dim = embedding.size()

    embedding = embedding.view(-1, embedding_dim)
    label = label.view(-1, one_hot_dim)

    # remove the loss of silence TF regions
    silence_mask = torch.sum(label, dim=-1, keepdim=True)
    embedding = silence_mask * embedding

    # referred as weight VA
    class_weights = F.normalize(torch.sum(label, dim=-2),
                                            p=1, dim=-1).unsqueeze(0)
    class_weights = 1.0 / (torch.sqrt(class_weights) + 1e-7)
    weights = torch.mm(label, class_weights.transpose(1, 0))
    label = label * weights.repeat(1, label.size()[-1])
    embedding = embedding * weights.repeat(1, embedding.size()[-1])

    # do batch affinity matrix computation
    embedding = embedding.view(batch_size,-1,embedding_dim)
    label = label.view(batch_size,-1,one_hot_dim)
    loss_est = norm(torch.bmm(T(embedding), embedding))
    loss_est_true = 2*norm(torch.bmm(T(embedding), label))
    loss_true = norm(torch.bmm(T(label), label))
    loss_embedding = loss_est - loss_est_true + loss_true

    return loss_embedding
