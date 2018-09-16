import torch
import torch.nn as nn

class DCNet(nn.Module):
    def __init__(
        self,input_size, hidden_size,embed_dim,num_layers):
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        super(DCNet, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout = 0.2, bidirectional=True,batch_first = True)
        self.fc = nn.Linear(hidden_size*2,embed_dim*input_size)

    def forward(self, x):
       #x is N*T*F tensor
       batch_size = x.shape[0]
       hidden = (
                torch.randn(self.num_layers*2, batch_size, self.hidden_size),
                torch.randn(self.num_layers*2, batch_size, self.hidden_size)
                )
       out, hidden = self.lstm(x,hidden)
       vec = self.fc(out)
       return vec

def loss_func(out,target,embed_dim):
    #out is N*T*(F*embed_dim)
    #reshape it to -1*embed_dim
    out =out.view(-1,12900,embed_dim)
    target = target.view(-1,12900,3)
    l = 0.0
    for i in range(out.shape[0]):
        A = torch.matmul(target[i], torch.t(target[i]))
        A_ = torch.matmul(out[i], torch.t(out[i]))
        l += torch.norm(A - A_,p=2)
    return l/(out.shape[0]*out.shape[1])

#TODO
#Add batch norm module at the first (or all) Module
#define the loss function in the paper
#Extract the feature and seg it to batch*100*513
#Train on 100 frame segs then 400 segs
