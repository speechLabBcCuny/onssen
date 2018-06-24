from model import DCNet, loss_func
import torch
import torch.nn as nn
import glob


embed_dim = 40
model = DCNet(129,32,100,embed_dim,2)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
num_epoch = 3

for epoch in num_epoch:
    f_list = glob.glob('/scratch/near/2speakers_stft/wav8k/min/tr/mix/*.mat')
    feat = None
    target = None
    model = MyNet(129,32,100,40,2)
    batch =0
    while len(f_list)>0 or feat is None or feat.shape[0]<3200:
        batch += 1
        feat, target = generate_samples(f_list, feat, target)
        inputs = feat[0:3200,:]
        labels = target[0:3200,:,:]
        inputs = inputs.reshape((32,100,129))
        labels = labels.reshape((32,100,129,3))
        inputs = Variable(torch.tensor(inputs,dtype=torch.float))
        labels = Variable(torch.tensor(labels,dtype=torch.long))
        feat = feat[3200:,:]
        target = target[3200:,:,:]
        out = model(inputs)
        optimizer.zero_grad()
        loss = loss_func(out.float(),labels.float(),40)
        loss.backward()
        if batch%100==0:
            print loss
        optimizer.step()

    torch.save(model,'%d_model'%epoch)
