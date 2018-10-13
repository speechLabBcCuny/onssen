from model import DCNet
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
        (_, inputs_train, labels_train) = generate_samples_chimera_net(train_list, BATCH_SIZE)
        inputs = Variable(torch.tensor(inputs_train, dtype=torch.float,device=cuda))
        labels = Variable(torch.tensor(labels_train, dtype=torch.float,device=cuda))
        # compute output
        out = model(inputs)
        loss = model.affinity_cost(out,labels)
        losses.update(torch.mean(loss), inputs.size(0))
        print('%d/%d, loss: %f'%(i+1,SAMPLE_PER_EPOCH,losses.avg), end='\r')
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        optimizer.step()
    stdout.write("\n")
    print("Training Loss: %f"%losses.avg)

def valid(valid_list, model, optimizer, epoch):
    losses = AverageMeter()
    # switch to train mode
    model.eval()
    with torch.no_grad():
        for i in range(SAMPLE_PER_EPOCH):
            (_, inputs_valid, labels_valid) = generate_samples_chimera_net(valid_list, BATCH_SIZE)
            inputs = Variable(torch.tensor(inputs_valid, dtype=torch.float,device=cuda))
            labels = Variable(torch.tensor(labels_valid, dtype=torch.float,device=cuda))
            # compute output
            out = model(inputs)
            loss = model.affinity_cost(out,labels)
            # measure record loss
            losses.update(torch.mean(loss), inputs.size(0))
    print("Validation Loss: %f"%losses.avg)
    return losses.avg


def main():
    train_list = glob.glob('/scratch/near/2speakers/wav8k/min/tr/mix/*.wav')
    valid_list = glob.glob('/scratch/near/2speakers/wav8k/min/cv/mix/*.wav')
    model = DCNet(129,RNN_SIZE,EMBEDDING_DIM,NUM_RNN_LAYERS)
    min_loss = float('inf')
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25,gamma=0.1)
    for epoch in range(NUM_EPOCH):
        train(train_list, model, optimizer, epoch)
        valid_loss = valid(valid_list, model, optimizer, epoch)
        scheduler.step()
        if valid_loss<min_loss:
            min_loss = valid_loss
            torch.save(model,'./dc_models/%d_%.2fmodel'%(epoch,valid_loss))

if __name__ == '__main__':
    main()
