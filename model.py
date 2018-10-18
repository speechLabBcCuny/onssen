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
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.C = C
        rnn = nn.LSTM(input_size,hidden_size,num_layers,dropout = 0.3, bidirectional=True,batch_first = True)
        fc_dc = nn.Linear(hidden_size*2,embed_dim*input_size)
        fc_mi = nn.Linear(hidden_size*2, C*input_size)
        sigmoid_dc = nn.Sigmoid()
        sigmoid_mi = nn.Sigmoid()
        self.add_module('rnn', rnn)
        self.add_module('fc_dc', fc_dc)
        self.add_module('fc_mi',fc_mi)
        self.add_module('sigmoid_dc',sigmoid_dc)
        self.add_module('sigmoid_mi',sigmoid_mi)
    def forward(self, x):
       #x is N*T*F tensor
       sequence_length = x.size(1)
       num_frequencies = x.size(2)
       batch_size = x.shape[0]
       #hidden = torch.randn(self.num_layers*2, batch_size, self.hidden_size)
       rnn_output, hidden = self.rnn(x)
       rnn_output = rnn_output.contiguous()
       rnn_output = rnn_output.view(-1, sequence_length, 2*self.hidden_size)
       embedding = self.fc_dc(rnn_output)
       mask = self.fc_mi(rnn_output)
       embedding = embedding.view(-1, sequence_length*num_frequencies, self.embed_dim)
       embedding = nn.functional.normalize(embedding, p=2, dim=-1)
       embedding = self.sigmoid_dc(embedding)
       mask = mask.view(-1,sequence_length,num_frequencies, self.C)
       mask = self.sigmoid_mi(mask)
       return (embedding,mask)

    @staticmethod
    def affinity_cost(embedding, assignments, mask, noisy_mag, clean_s1, clean_s2, alpha):
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
        loss_embedding = loss_est - loss_est_true + loss_true
        #permutation invariant training
        loss_s1 = torch.min(norm(mask[:,:,:,0]*noisy_mag - clean_s1), norm(mask[:,:,:,1]*noisy_mag - clean_s1))
        loss_s2 = torch.min(norm(mask[:,:,:,0]*noisy_mag - clean_s2), norm(mask[:,:,:,1]*noisy_mag - clean_s2))
        loss_mask = loss_s1+loss_s2
        #loss = loss / (loss_est + loss_true)
        return loss_embedding*alpha + loss_mask*(1-alpha)
