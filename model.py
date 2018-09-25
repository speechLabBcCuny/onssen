import torch
import torch.nn as nn
from torch.autograd import Variable

class DCNet(nn.Module):
    def __init__(
        self,input_size, hidden_size,embed_dim,num_layers):
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        super(DCNet, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout = 0.2, bidirectional=True,batch_first = True)
        self.fc = nn.Linear(hidden_size*2,embed_dim*input_size)
        self.tanh = nn.Tanh()
    def forward(self, x):
       #x is N*T*F tensor
       batch_size = x.shape[0]
       hidden = (
                torch.randn(self.num_layers*2, batch_size, self.hidden_size),
                torch.randn(self.num_layers*2, batch_size, self.hidden_size)
                )
       out, hidden = self.lstm(x,hidden)
       vec = self.fc(out)
       vec_tanh = self.tanh(vec)
       return vec_tanh

def loss_func(V,Y,embed_dim):
    #out is N*T*(F*embed_dim)
    #reshape it to -1*embed_dim
    V =V.reshape(-1,embed_dim)
    Y = Y.reshape(-1,3)
    index = []
    for i in range(Y.shape[0]):
        if not Y[i,2]==1:
            index.append(i)
    index = torch.tensor(index,dtype=torch.long)
    V = torch.index_select(V,0,index)
    Y = torch.index_select(Y,0,index)
    l = 0.0
    I = torch.ones((1,Y.shape[0]), dtype=torch.float)
    D = torch.matmul(Y,torch.t(torch.matmul(I,Y)))
    D_sqrt = 1/torch.sqrt(D)
    D_sqrt = D_sqrt.reshape(D_sqrt.shape[0])
    l = 0.0
    l += torch.norm(torch.matmul(torch.t(V)*D_sqrt,V),p=2)
    l -=2*torch.norm(torch.matmul(torch.t(V)*D_sqrt,Y),p=2)
    l += torch.norm(torch.matmul(torch.t(Y)*D_sqrt,Y),p=2)
    return l

# def loss_func(out,target,embed_dim):
#     #out is N*T*(F*embed_dim)
#     #reshape it to -1*embed_dim
#     out =out.view(-1,12900,embed_dim)
#     target = target.view(-1,12900,3)
#     l = 0.0
#     for i in range(out.shape[0]):
#         A = torch.matmul(target[i], torch.t(target[i]))
#         A_ = torch.matmul(out[i], torch.t(out[i]))
#         l += torch.norm(A - A_,p=2)
#     return l/(out.shape[0]*out.shape[1])
#
# #TODO
#Add batch norm module at the first (or all) Module
#Train on 100 frame segs then 400 segs


class ChimeraNet(nn.Module):
    def __init__(
        self,input_size, hidden_size,embed_dim,num_layers,C):
        super(ChimeraNet, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.C = C
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout = 0.2, bidirectional=True,batch_first = True)
        self.fc = nn.Linear(hidden_size*2,embed_dim*input_size)
        self.tanh = nn.Tanh()
        self.mask = nn.Linear(embed_dim, self.C)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        #x is N*T*F tensor
        batch_size = x.shape[0]
        hidden = (
                torch.randn(self.num_layers*2, batch_size, self.hidden_size),
                torch.randn(self.num_layers*2, batch_size, self.hidden_size)
                )
        out, hidden = self.lstm(x,hidden)
        vec = self.fc(out)
        vec_tanh = self.tanh(vec)
        vec = vec.reshape(vec.shape[0],vec.shape[1],self.input_size,self.embed_dim)
        mask = self.mask(vec)
        mask = self.softmax(mask)
        return (vec_tanh, mask)

def chi_loss_func(inputs, result,Y,embed_dim, alpha):
    #out is N*T*(F*embed_dim)
    #reshape it to -1*embed_dim
    mask_label = Y[:,:,:,0:2]
    V = result[0]
    V = V.float()
    Mask = result[1]
    V =V.reshape(-1,embed_dim)
    Y = Y.reshape(-1,3)
    index = []
    for i in range(Y.shape[0]):
        if not Y[i,2]==1:
            index.append(i)
    index = torch.tensor(index,dtype=torch.long)
    V = torch.index_select(V,0,index)
    Y = torch.index_select(Y,0,index)
    l = 0.0
    I = torch.ones((1,Y.shape[0]), dtype=torch.float)
    D = torch.matmul(Y,torch.t(torch.matmul(I,Y)))
    D_sqrt = 1/torch.sqrt(D)
    D_sqrt = D_sqrt.reshape(D_sqrt.shape[0])
    l = 0.0
    l += torch.norm(torch.matmul(torch.t(V)*D_sqrt,V),p=2)
    l -=2*torch.norm(torch.matmul(torch.t(V)*D_sqrt,Y),p=2)
    l += torch.norm(torch.matmul(torch.t(Y)*D_sqrt,Y),p=2)
    inputs = inputs.reshape(inputs.shape+(1,))
    l_mask = torch.norm(torch.mul(inputs,(mask_label-Mask)),p=2)
    return l*alpha+l_mask*(1-alpha)
