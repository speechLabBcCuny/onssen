import glob, numpy as np, torch
from sklearn.model_selection import train_test_split


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_train_validation_list(data_path, train_size=0.8):
    file_list = glob.glob(data_path+'*.wav')
    file_list = np.array(file_list)
    train, validation = train_test_split(filenames,train_size=train_size)
    
