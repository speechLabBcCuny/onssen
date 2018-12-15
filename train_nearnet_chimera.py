from model import NearNet_mag_net, NearNet_phase_net, NearNet, loss_NearNet
from config import *
#from chimera_feats import get_chimera_samples, get_speaker_dict
from feat_generator import generate_samples_near_net_chimera
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import time
import os
from data_generator import WSJ0_2Mix_Dataset
from torch.utils.data import DataLoader

#modify the config
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

def train(train_loader, mode, model, optimizer, epoch, alpha, cuda):
    losses = AverageMeter()
    times = AverageMeter()
    losses.reset()
    times.reset()
    # switch to train mode
    model.train()
    for i, data in enumerate(train_loader):
        begin = time.time()
        d = [ele for ele in data]
        for j in range(1, len(d)):
            d[j] = Variable(d[j].float().cuda())
        d[0] = Variable(d[0].float().cuda(), requires_grad = True)
        d[2] = Variable(d[2].float().cuda(), requires_grad = True)
        d[3] = Variable(d[3].float().cuda(), requires_grad = True)
        [feat,target,noisy_mag,noisy_phase,s1_mag,s2_mag,s1_phase,s2_phase] = d
        # compute output
        (embedding, mask_A, mask_B, phase_A, phase_B) = model(feat, noisy_mag, noisy_phase)
        loss = loss_NearNet(embedding, target, noisy_mag,
                        mask_A, mask_B, phase_A, phase_B,
                        s1_mag, s2_mag,
                        s1_phase, s2_phase,
                        alpha)
        #loss = model.loss_NearNet(mask, phase, labels_train, noisy_mag, clean_s1, clean_s2, alpha)
        avg_loss = torch.mean(loss)
        losses.update(avg_loss.item(), feat.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        end = time.time()
        times.update(end-begin)
        print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len(train_loader),losses.avg, times.avg*len(train_loader)), end='\r')
    print("\n")

def valid(valid_loader, mode, model, optimizer, epoch, alpha, cuda):
    losses = AverageMeter()
    times = AverageMeter()
    losses.reset()
    times.reset()
    # switch to train mode
    model.eval()
    with torch.no_grad():
        for i in range(50):
            begin = time.time()
            d = [ele for ele in next(iter(valid_loader))]
            for j in range(1, len(d)):
                d[j] = Variable(d[j].float().cuda())
            d[0] = Variable(d[0].float().cuda(), requires_grad = True)
            d[2] = Variable(d[2].float().cuda(), requires_grad = True)
            d[3] = Variable(d[3].float().cuda(), requires_grad = True)
            [feat,target,noisy_mag,noisy_phase,s1_mag,s2_mag,s1_phase,s2_phase] = d
            feat = Variable(feat.float().cuda(),requires_grad=True)
            target = Variable(target.float().cuda())
            noisy_mag = Variable(noisy_mag.float().cuda())
            noisy_phase = Variable(noisy_phase.float().cuda())
            s1_mag = Variable(s1_mag.float().cuda())
            s2_mag = Variable(s2_mag.float().cuda())
            s1_phase = Variable(s1_phase.float().cuda())
            s2_phase = Variable(s2_phase.float().cuda())
            (embedding, mask_A, mask_B, phase_A, phase_B) = model(feat, noisy_mag, noisy_phase)
            loss = loss_NearNet(embedding, target, noisy_mag,
                            mask_A, mask_B, phase_A, phase_B,
                            s1_mag, s2_mag,
                            s1_phase, s2_phase,
                            alpha)
            avg_loss = torch.mean(loss)
            losses.update(avg_loss.item(), feat.size(0))
            end = time.time()
            times.update(end-begin)
            print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds'%(epoch, i+1,50,losses.avg, times.avg*50), end='\r')
        print("\n")
    return losses.avg


def main():
    #use cuda
    cuda = torch.device('cuda')

    train_list = glob.glob('/scratch/near/2speakers/wav8k/min/tr/mix/*.wav')
    valid_list = glob.glob('/scratch/near/2speakers/wav8k/min/cv/mix/*.wav')
    #model = torch.load('./nearnet_models_chimera/model_12_02/nearnet_alpha_0.700000_epoch_39_-98.08model')
    model = NearNet(129, 600, 4, 20, 2)
    # model.fc_mask_real.weight.requires_grad=True
    # model.fc_mask_real.bias.requires_grad=True
    # model.fc_mask_imag.weight.requires_grad=True
    # model.fc_mask_imag.bias.requires_grad=True
    # model.fc_dc.weight.requires_grad=False
    # model.fc_dc.bias.requires_grad=False
    alpha = 0.9
    min_loss_mask = float('inf')
    model = nn.DataParallel(model)
    model.to(cuda)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,betas=[0.9,0.999])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25,gamma=0.1)
    count_mask = 0
    train_loader =DataLoader(WSJ0_2Mix_Dataset('tr'), batch_size=16, shuffle=True, num_workers=2)
    valid_loader =DataLoader(WSJ0_2Mix_Dataset('cv'), batch_size=16, shuffle=True, num_workers=2)
    for epoch in range(0,NUM_EPOCH):
        train(train_loader, 'train',model, optimizer, epoch, alpha, cuda)
        valid_loss = valid(valid_loader, 'valid', model, optimizer, epoch, alpha, cuda)
        #scheduler.step()
        if valid_loss<min_loss_mask:
            count_mask = 0
            min_loss_mask = valid_loss
            model_path = './nearnet_models_chimera/model_12_12/nearnet_alpha_%f_epoch_%d_%.2fmodel'%(alpha,epoch,valid_loss)
            directory = os.path.dirname(model_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model,model_path)
        else:
            count_mask+=1
            if count_mask == 10:
                break
if __name__ == '__main__':
    main()
