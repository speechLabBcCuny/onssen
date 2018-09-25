from model import DCNet, loss_func
from feat_generator import generate_samples
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import time
batch_size=32
embed_dim = 20
num_layers = 4
hidden_size = 500
model = DCNet(129,hidden_size,embed_dim,num_layers)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
num_epoch = 10

for epoch in range(num_epoch):
    f_list = glob.glob('/scratch/near/2speakers_stft/wav8k/min/tr/mix/*.mat')
    feat = None
    target = None
    batch =0
    while len(f_list)>0 and (feat is None or feat.shape[0]<batch_size*100):
        batch += 1
        print("batch: %d"%batch)
        feat, target = generate_samples(f_list, feat, target)
        inputs = feat[0:batch_size*100,:]
        labels = target[0:batch_size*100,:,:]
        if inputs.shape[0]<batch_size*100:
            break
        inputs = inputs.reshape((batch_size,100,129))
        labels = labels.reshape((batch_size,100,129,3))
        inputs = Variable(torch.tensor(inputs,dtype=torch.float))
        labels = Variable(torch.tensor(labels,dtype=torch.long))
        feat = feat[batch_size*100:,:]
        target = target[batch_size*100:,:,:]
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_func(out.float(),labels.float(), embed_dim)
        loss.backward()
        optimizer.step()
        print("loss: %f"%loss)
    torch.save(model,'./dc_models/%d_model'%epoch)
