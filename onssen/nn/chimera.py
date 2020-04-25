import torch
import torch.nn as nn
import torch.nn.functional as F

class chimera(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=300,
        num_layers=3,
        embedding_dim=20,
        dropout=0.3,
        num_speaker=2,
    ):
        super(chimera, self).__init__()
        rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first = True
        )
        bn = nn.BatchNorm1d(hidden_dim*2)
        fc_dc = nn.Linear(hidden_dim*2, input_dim*embedding_dim)
        fc_mi = nn.Linear(hidden_dim*2, input_dim*num_speaker)
        self.add_module('rnn', rnn)
        self.add_module('bn', bn)
        self.add_module('fc_dc', fc_dc)
        self.add_module('fc_mi', fc_mi)

    def forward(self, input):
        assert len(input)==1, "There must be one tensor in the input for the chimera network"
        # x is N*T*F tensor
        x = input[0].float()
        batch_size, frame, frequency = x.size()
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(x)
        rnn_output = rnn_output.permute(0, 2, 1)
        rnn_output = self.bn(rnn_output)
        rnn_output = rnn_output.permute(0, 2, 1)
        embedding = self.fc_dc(rnn_output)
        masks = self.fc_mi(rnn_output)
        embedding = embedding.reshape(batch_size, frame*frequency, -1)
        embedding = F.normalize(embedding, p=2, dim=-1)
        embedding = embedding.reshape(batch_size, frame, frequency, -1)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(batch_size, frame, frequency, -1)
        mask_A = masks[:,:,:,0]
        mask_B = masks[:,:,:,1]
        return [embedding, mask_A, mask_B]
