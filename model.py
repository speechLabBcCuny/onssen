import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
cuda = torch.device('cuda')

class DCNet(nn.Module):
    def __init__(
        self,input_size, hidden_size,embed_dim,num_layers):
        super(DCNet, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn = nn.LSTM(input_size,hidden_size,num_layers,dropout = 0.5, bidirectional=True,batch_first = True)
        fc = nn.Linear(hidden_size*2,embed_dim*input_size)
        self.add_module('rnn', rnn)
        self.add_module('fc', fc)
    def forward(self, x):
       #x is N*T*F tensor
       sequence_length = x.size(1)
       num_frequencies = x.size(2)
       batch_size = x.shape[0]
       #hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size)
       output, hidden = self.rnn(x)
       output = output.contiguous()
       output = output.view(-1, sequence_length, 2*self.hidden_size)
       embedding = self.fc(output)
       embedding = embedding.view(-1, sequence_length*num_frequencies, self.embed_dim)
       embedding = nn.functional.normalize(embedding, p=2, dim=-1)
       return embedding

    @staticmethod
    def affinity_cost(embedding, assignments):
        """
        Function defining the affinity cost for deep clustering
        Args:
            embedding:
            assignments:
        Returns:
        """
        batch_size = embedding.size()[0]
        embedding_dim = embedding.size()[-1]
        one_hot_dim = assignments.size()[-1]
        def T(tensor):
            return tensor.permute(0,2,1)
        def norm(tensor):
            tensor_sq = torch.mul(tensor,tensor)
            tensor_sq = tensor_sq.view(batch_size, -1)
            return torch.sum(tensor_sq,dim=1)
        embedding = embedding.view(-1, embedding.size()[-1])
        assignments = assignments.view(-1, assignments.size()[-1])
        silence_mask = torch.sum(assignments, dim=-1, keepdim=True)
        embedding = silence_mask * embedding

        class_weights = nn.functional.normalize(torch.sum(assignments, dim=-2),
                                                p=1, dim=-1).unsqueeze(0)
        class_weights = 1.0 / (torch.sqrt(class_weights) + 1e-7)
        weights = torch.mm(assignments, class_weights.transpose(1, 0))
        assignments = assignments * weights.repeat(1, assignments.size()[-1])
        embedding = embedding * weights.repeat(1, embedding.size()[-1])
        embedding = embedding.view(batch_size,-1,embedding_dim)
        assignments = assignments.view(batch_size,-1,one_hot_dim)

        loss_est = norm(torch.bmm(T(embedding), embedding))
        loss_est_true = 2*norm(torch.bmm(T(embedding), assignments))
        loss_true = norm(torch.bmm(T(assignments), assignments))
        loss = loss_est - loss_est_true + loss_true
        #loss = loss / (loss_est + loss_true)
        return loss


# def loss_func(V,Y,embed_dim):
#     #out is N*T*(F*embed_dim)
#     #reshape it to -1*embed_dim
#     V =V.reshape(-1,embed_dim)
#     Y = Y.reshape(-1,3)
#     index = []
#     for i in range(Y.shape[0]):
#         if not Y[i,2]==1:
#             index.append(i)
#     index = torch.tensor(index,dtype=torch.long)
#     V = torch.index_select(V,0,index)
#     Y = torch.index_select(Y,0,index)
#     l = 0.0
#     I = torch.ones((1,Y.shape[0]), dtype=torch.float)
#     D = torch.matmul(Y,torch.t(torch.matmul(I,Y)))
#     D_sqrt = 1/torch.sqrt(D)
#     D_sqrt = D_sqrt.reshape(D_sqrt.shape[0])
#     l = 0.0
#     l += torch.norm(torch.matmul(torch.t(V)*D_sqrt,V),p=2)
#     l -=2*torch.norm(torch.matmul(torch.t(V)*D_sqrt,Y),p=2)
#     l += torch.norm(torch.matmul(torch.t(Y)*D_sqrt,Y),p=2)
#     return l

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

# #TODO
#Add batch norm module at the first (or all) Module
#Train on 100 frame segs then 400 segs

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class ChimeraNet(nn.Module):
    def __init__(
        self,input_size, hidden_size,embed_dim,num_layers,C):
        super(ChimeraNet, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.C = C
        self.bn_models = [SequenceWise(nn.BatchNorm1d(input_size))]
        self.lstm_models = [nn.GRU(input_size,hidden_size, bidirectional=True,batch_first = True,bias=True)]
        for i in range(num_layers-1):
            self.bn_models.append(SequenceWise(nn.BatchNorm1d(hidden_size*2)))
        for i in range(num_layers-1):
            self.lstm_models.append(nn.GRU(hidden_size*2,hidden_size,bidirectional=True,batch_first = True,bias=True))
        for module in self.lstm_models:
            self.add_module('lstm',module)
        for module in self.bn_models:
            self.add_module('bn',module)
        #self.lstm = nn.LSTM(input_size,hidden_size,num_layers, bidirectional=True,batch_first = True,bias=True)
        self.fc = nn.Linear(hidden_size*2,embed_dim*input_size)
        self.tanh = nn.Tanh()
        self.mask = nn.Linear(embed_dim, self.C)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        #x is N*T*F tensor
        batch_size = x.shape[0]
        for i in range(self.num_layers):
            x = self.bn_models[i](x)
            hidden = torch.randn(2, batch_size, self.hidden_size)
            x, hidden = self.lstm_models[i](x, hidden)
        vec = self.fc(x)
        vec = vec.reshape(vec.shape[:-1]+(self.input_size,self.embed_dim))
        vec_tanh = self.tanh(vec)
        vec_norm = f.normalize(vec_tanh, dim=3,p=2)
        mask = self.mask(vec)
        mask = self.softmax(mask)
        return (vec_norm, mask)

class chi_loss_func(torch.nn.Module):
    def __init__(self):
        super(chi_loss_func,self).__init__()

    def forward(self,inputs, result,Y,embed_dim, alpha):
        #out is N*T*(F*embed_dim)
        #reshape it to -1*embed_dim
        mask_label = Y[:,:,:,0:2]
        V = result[0]
        V = V.float()
        Mask = result[1]
        V =V.view(-1,embed_dim)
        Y = Y.view(-1,3)
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
        l += torch.norm(torch.matmul(torch.t(V)*D_sqrt,V))
        l -=2*torch.norm(torch.matmul(torch.t(V)*D_sqrt,Y))
        l += torch.norm(torch.matmul(torch.t(Y)*D_sqrt,Y))
        inputs = inputs.reshape(inputs.shape+(1,))
        l_mask = torch.norm(torch.mul(inputs,(mask_label-Mask)))
        return (l*alpha+l_mask*(1-alpha))
