from model import DCNet, loss_func
from feat_generator import generate_samples
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import time

embed_dim = 40
model = DCNet(129,100,embed_dim,2)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
num_epoch = 3

for epoch in range(num_epoch):
    f_list = glob.glob('/scratch/near/2speakers_stft/wav8k/min/tr/mix/*.mat')
    feat = None
    target = None
    batch =0
    while len(f_list)>0 and (feat is None or feat.shape[0]<3200):
        batch += 1
        print("batch: %d"%batch)
        feat, target = generate_samples(f_list, feat, target)
        inputs = feat[0:3200,:]
        labels = target[0:3200,:,:]
        if inputs.shape[0]<3200:
            break
        inputs = inputs.reshape((32,100,129))
        labels = labels.reshape((32,100,129,3))
        inputs = Variable(torch.tensor(inputs,dtype=torch.float))
        labels = Variable(torch.tensor(labels,dtype=torch.long))
        feat = feat[3200:,:]
        target = target[3200:,:,:]
        out = model(inputs)
        optimizer.zero_grad()
        loss = loss_func(out.float(),labels.float(),inputs, 40)
        loss.backward()
        if batch%10==0:
            print("loss: %f"%loss)
        optimizer.step()
    torch.save(model,'./dc_models/%d_model'%epoch)
