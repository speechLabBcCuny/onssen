from model import ChimeraNet, chi_loss_func
from feat_generator import generate_samples_chimera_net
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import time

embed_dim = 20
num_layers = 4
hidden_size = 500
C = 2
model = ChimeraNet(129,hidden_size,embed_dim,num_layers,C)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
num_epoch = 10
batch_size = 32
for epoch in range(num_epoch):
    f_list = glob.glob('/scratch/near/2speakers_stft/wav8k/min/tr/mix/*.mat')
    magnitude=None
    feat = None
    target = None
    batch =0
    while (len(f_list)>0 or feat is None or feat.shape[0]<batch_size*100):
        batch += 1
        print("batch: %d"%batch)
        magnitude, feat, target = generate_samples_chimera_net(f_list, batch_size, magnitude, feat, target)
        inputs = feat[0:batch_size*100,:]
        labels = target[0:batch_size*100,:,:]
        mag = magnitude[0:batch_size*100,:]
        mag = mag.reshape((batch_size,100,129))
        inputs = inputs.reshape((batch_size,100,129))
        labels = labels.reshape((batch_size,100,129,3))
        mag = Variable(torch.tensor(mag, dtype=torch.float))
        inputs = Variable(torch.tensor(inputs, dtype=torch.float))
        labels = Variable(torch.tensor(labels, dtype=torch.long))
        magnitude = magnitude[batch_size*100:,:]
        feat = feat[batch_size*100:,:]
        target = target[batch_size*100:,:,:]
        optimizer.zero_grad()
        out = model(inputs)
        loss = chi_loss_func(mag, out,labels.float(),embed_dim,0.5)
        loss.backward()
        optimizer.step()
        if batch%10==0:
            print("loss: %f"%loss)
    torch.save(model,'./chimera_models/%d_model'%epoch)
