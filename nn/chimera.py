import torch
import torch.nn as nn
import torch.nn.functional as F

class chimera(nn.Module):
    def __init__(self, model_options):
        super(chimera, self).__init__()
        self.input_dim = model_options.input_dim
        self.output_dim = model_options.output_dim if "output_dim" in model_options else self.input_dim
        self.hidden_dim = model_options.hidden_dim if "hidden_dim" in model_options else 300
        self.num_layers = model_options.num_layers if "num_layers" in model_options else 3
        self.embedding_dim = model_options.embedding_dim if "embedding_dim" in model_options else 20
        self.dropout = model_options.dropout if "dropout" in model_options else 0.3
        self.num_speaker = model_options.num_speaker if "num_speaker" in model_options else 2
        rnn = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first = True
        )
        bn = nn.BatchNorm1d(self.hidden_dim*2)
        fc_dc = nn.Linear(self.hidden_dim * 2, self.output_dim * self.embedding_dim)
        fc_mi = nn.Linear(self.hidden_dim * 2, self.output_dim * self.num_speaker)
        self.add_module('rnn', rnn)
        self.add_module('bn', bn)
        self.add_module('fc_dc', fc_dc)
        self.add_module('fc_mi', fc_mi)

    def forward(self, input):
        # x is N*T*F tensor
        x = input[0].float()
        batch_size, frame_size, _ = x.size()
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(x)
        rnn_output = rnn_output.permute(0, 2, 1)
        rnn_output = self.bn(rnn_output)
        rnn_output = rnn_output.permute(0, 2, 1)
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
