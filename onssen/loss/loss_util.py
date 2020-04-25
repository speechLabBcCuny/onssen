import torch


def T(tensor):
    return tensor.permute(0, 2, 1)

def norm(tensor):
    batch_size = tensor.size()[0]
    tensor_sq = torch.mul(tensor, tensor)
    tensor_sq = tensor_sq.view(batch_size, -1)
    return torch.sqrt(torch.sum(tensor_sq, dim=1))

def norm_1d(tensor):
    batch_size = tensor.size()[0]
    tensor = tensor.reshape(batch_size, -1)
    return torch.sum(torch.abs(tensor), dim=1)
