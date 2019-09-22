import torch.nn as nn
import torch.nn.functional as F

class deep_clustering(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=300,
        embedding_dim=20,
        num_layers=3,
        dropout=0.3,
    ):
        super(deep_clustering, self).__init__()
        rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first = True
        )
        fc_dc = nn.Linear(hidden_dim*2, embedding_dim*input_dim)
        self.add_module('rnn', rnn)
        self.add_module('fc_dc', fc_dc)

    def forward(self, input):
       # x is N*T*F tensor
       assert len(input)==1, "There must be one tensor in the input for Deep Clustering Model"
       x = input[0].float()
       batch_size, frame_size, frequency_size = x.size()
       self.rnn.flatten_parameters()
       rnn_output, hidden = self.rnn(x)
       embedding = self.fc_dc(rnn_output)
       embedding = embedding.view(batch_size, frame_size*frequency_size, -1)
       embedding = F.normalize(embedding, p=2, dim=-1)
       embedding = embedding.reshape(batch_size, frame_size, frequency_size, -1)
       return [embedding]
