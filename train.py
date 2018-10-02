from model import DCNet
from feat_generator import generate_samples_chimera_net
import torch
import torch.nn as nn
import glob
from torch.autograd import Variable
import time

#use cuda
cuda = torch.device('cuda')

batch_size=32
embed_dim = 20
num_layers = 4
hidden_size = 500
num_epoch = 10
with torch.cuda.device(0):
    model = DCNet(129,hidden_size,embed_dim,num_layers)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epoch):
        f_list = glob.glob('/scratch/near/2speakers_stft/wav8k/min/tr/mix/*.mat')
        magnitude = None
        feat = None
        target = None
        batch =0
        while len(f_list)>0 and (feat is None or feat.shape[0]<batch_size*100):
            batch += 1
            print("batch: %d"%batch)
            (mag, inputs, labels),(magnitude, feat, target) = generate_samples_chimera_net(f_list, batch_size, magnitude, feat, target)
            if inputs is None:
                break
            #mag = Variable(torch.tensor(mag, dtype=torch.float,device=cuda))
            inputs = Variable(torch.tensor(inputs, dtype=torch.float,device=cuda))
            labels = Variable(torch.tensor(labels, dtype=torch.float,device=cuda))
            out = model(inputs)
            optimizer.zero_grad()
            out = model(inputs)
            loss = model.affinity_cost(out,labels)
            loss.backward()
            optimizer.step()
            print("loss: %f"%loss)
        torch.save(model,'./dc_models/%d_model'%epoch)
