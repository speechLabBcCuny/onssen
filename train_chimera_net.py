from model import ChimeraNet
from config import *
from feat_generator import generate_samples_chimera_net
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import time
from sys import stdout

#use cuda
cuda = torch.device('cuda')

#modify the config
NUM_RNN_LAYERS=4
NUM_FRAME=400
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

def train(train_list, model, optimizer, epoch):
    losses = AverageMeter()
    # switch to train mode
    model.train()
    for i in range(SAMPLE_PER_EPOCH):
        (noisy_mag, clean_s1, clean_s2, inputs_train, labels_train) = generate_samples_chimera_net(train_list, BATCH_SIZE, NUM_FRAME)
        noisy_mag = torch.from_numpy(noisy_mag).float().to(cuda)
        inputs_train = torch.from_numpy(inputs_train).float().to(cuda)
        labels_train = torch.from_numpy(labels_train).float().to(cuda)
        clean_s1 = torch.from_numpy(clean_s1).float().to(cuda)
        clean_s2 = torch.from_numpy(clean_s2).float().to(cuda)
        # compute output
        (embedding, mask) = model(inputs_train)
        loss = model.affinity_cost(embedding,labels_train, mask, noisy_mag, clean_s1, clean_s2, CHIMERA_ALPHA)
        losses.update(torch.mean(loss), inputs_train.size(0))
        print('%d/%d, training loss: %f'%(i+1,SAMPLE_PER_EPOCH,losses.avg), end='\r')
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        optimizer.step()
    print("\n")

def valid(valid_list, model, optimizer, epoch):
    losses = AverageMeter()
    # switch to train mode
    model.eval()
    with torch.no_grad():
        for i in range(SAMPLE_PER_EPOCH):
            (noisy_mag, clean_s1, clean_s2, inputs_valid, labels_valid) = generate_samples_chimera_net(valid_list, BATCH_SIZE, NUM_FRAME)
            noisy_mag = torch.from_numpy(noisy_mag).float().to(cuda)
            inputs_valid = torch.from_numpy(inputs_valid).float().to(cuda)
            labels_valid = torch.from_numpy(labels_valid).float().to(cuda)
            clean_s1 = torch.from_numpy(clean_s1).float().to(cuda)
            clean_s2 = torch.from_numpy(clean_s2).float().to(cuda)
            # compute output
            (embedding, mask) = model(inputs_valid)
            loss = model.affinity_cost(embedding,labels_valid, mask, noisy_mag, clean_s1, clean_s2, CHIMERA_ALPHA)
            # measure record loss
            losses.update(torch.mean(loss), inputs.size(0))
            print('%d/%d, validation loss: %f'%(i+1,SAMPLE_PER_EPOCH,losses.avg), end='\r')
        print("\n")
    return losses.avg


def main():
    train_list = glob.glob('/scratch/near/2speakers/wav8k/min/tr/mix/*.wav')
    valid_list = glob.glob('/scratch/near/2speakers/wav8k/min/cv/mix/*.wav')
    model = ChimeraNet(129,RNN_SIZE,EMBEDDING_DIM,NUM_RNN_LAYERS,NUM_MASK)
    min_loss = float('inf')
    model.to(cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25,gamma=0.1)
    count = 0
    for epoch in range(NUM_EPOCH):
        train(train_list, model, optimizer, epoch)
        valid_loss = valid(valid_list, model, optimizer, epoch)
        scheduler.step()
        if valid_loss<min_loss:
            count = 0
            min_loss = valid_loss
            torch.save(model,'./chimera_models/%d_%.2fmodel'%(epoch,valid_loss))
        else:
            count+=1
            if count == 5:
                break
if __name__ == '__main__':
    main()
