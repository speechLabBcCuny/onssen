from model import ChimeraNet, chi_loss_func
from feat_generator import generate_samples_chimera_net
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

embed_dim = 20
num_layers = 4
hidden_size = 500
C = 2
model = ChimeraNet(129,hidden_size,embed_dim,num_layers,C)
criterion = chi_loss_func()
model.train()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
optimizer.zero_grad()
num_epoch = 10
batch_size = 32
for epoch in range(num_epoch):
    train_list = glob.glob('/scratch/near/2speakers_stft/wav8k/min/tr/mix/*.mat')
    magnitude=None
    feat = None
    target = None
    batch =0
    while (len(train_list)>0 or feat is None or feat.shape[0]<batch_size*100):
        batch += 1
        print("batch: %d"%batch)
        (mag, inputs, labels),(magnitude, feat, target) = generate_samples_chimera_net(train_list, batch_size, magnitude, feat, target)
        #(cv_mag, cv_inputs, cv_labels),(cv_magnitude, cv_feat, cv_target) = generate_samples_chimera_net(cv_list, batch_size, cv_magnitude, cv_feat, cv_target)
        # if inputs.shape[0]<batch_size*100:
        #     break
        if inputs is None:
            break
        mag = Variable(torch.tensor(mag, dtype=torch.float))
        inputs = Variable(torch.tensor(inputs, dtype=torch.float))
        labels = Variable(torch.tensor(labels, dtype=torch.float))
        out = model(inputs)
        loss = criterion(mag, out,labels,embed_dim,0.5)
        #loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if batch%1==0:
        print("loss: %f"%loss)
    torch.save(model,'./chimera_models/%d_model'%epoch)
