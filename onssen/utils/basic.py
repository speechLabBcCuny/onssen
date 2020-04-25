import glob, numpy as np, torch
from sklearn.model_selection import train_test_split


def build_optimizer(params, optimizer_options):
    if optimizer_options.name == "adam":
        return torch.optim.Adam(params, lr=optimizer_options.lr)
    if optimizer_options.name == "sgd":
        return torch.optim.SGD(params, lr=optimizer_options.lr, momentum=0.9)
    if optimizer_options.name == "rmsprop":
        return torch.optim.RMSprop(params, lr=optimizer_options.lr)

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

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def generate_train_validation_list(data_path, train_size=0.8):
    file_list = glob.glob(data_path+'*.wav')
    file_list = np.array(file_list)
    train, validation = train_test_split(filenames,train_size=train_size)
