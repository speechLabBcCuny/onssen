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

def loss_func(V,Y,embed_dim):
    #out is N*T*(F*embed_dim)
    #reshape it to -1*embed_dim
    V =V.view(-1,embed_dim)
    Y = Y.view(-1,3)
    I = torch.ones((1,Y.shape[0]), dtype=torch.float)
    D = torch.matmul(Y,torch.t(torch.matmul(I,Y)))
    D_sqrt = 1/torch.sqrt(D)
    D_sqrt = D_sqrt.reshape(D_sqrt.shape[0])
    l =0.0
    l += torch.norm(torch.matmul(torch.t(V)*D_sqrt,V),p=2)
    l -=2*torch.norm(torch.matmul(torch.t(V)*D_sqrt,Y),p=2)
    l += torch.norm(torch.matmul(torch.t(Y)*D_sqrt,Y),p=2)
    return l/Y.shape[0]

#TODO
#Add batch norm module at the first (or all) Module
#Train on 100 frame segs then 400 segs
