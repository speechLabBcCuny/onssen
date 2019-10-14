import torch
import torch.nn as nn
import torch.nn.functional as F

class chimera(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=300,
        embedding_dim=20,
        num_layers=3,
        dropout=0.5,
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
        fc_dc = nn.Linear(hidden_dim * 2, output_dim * embedding_dim)
        fc_mi = nn.Linear(hidden_dim * 2, output_dim * num_speaker)
        self.output_dim = output_dim
        self.add_module('rnn', rnn)
        self.add_module('fc_dc', fc_dc)
        self.add_module('fc_mi', fc_mi)

    def forward(self, input):
        # x is N*T*F tensor
        x = input[0].float()
        batch_size, frame_size, _ = x.size()
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(x)
        embedding = self.fc_dc(rnn_output)
        masks = self.fc_mi(rnn_output)
        embedding = embedding.reshape(batch_size, frame_size*self.output_dim, -1)
        embedding = F.normalize(embedding, p=2, dim=-1)
        embedding = embedding.reshape(batch_size, frame_size, self.output_dim, -1)
        masks = torch.sigmoid(masks)
        masks = masks.reshape(batch_size, frame_size, self.output_dim, -1)
        mask_A = masks[:,:,:,0]
        mask_B = masks[:,:,:,1]
        return [embedding, mask_A, mask_B]
