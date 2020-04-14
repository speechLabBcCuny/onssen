import torch
import torch.nn as nn
import torch.nn.functional as F
from .chimera import chimera


class phase_net(nn.Module):
    def __init__(self, model_options):
        super(phase_net, self).__init__()
        self.input_dim = model_options.input_dim
        self.output_dim = model_options.output_dim if "output_dim" in model_options else self.input_dim
        self.hidden_dim = model_options.hidden_dim if "hidden_dim" in model_options else 300
        self.num_layers = model_options.num_layers if "num_layers" in model_options else 3
        self.embedding_dim = model_options.embedding_dim if "embedding_dim" in model_options else 20
        self.dropout = model_options.dropout if "dropout" in model_options else 0.3
        self.num_speaker = model_options.num_speaker if "num_speaker" in model_options else 2
        chimera_net = chimera(model_options)
        rnn = nn.LSTM(
            self.input_dim*3,
            self.hidden_dim,
            self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first = True
        )
        bn = nn.BatchNorm1d(self.hidden_dim*2)
        fc_phase = nn.Linear(self.hidden_dim*2, self.num_speaker*self.output_dim)
        self.add_module('rnn', rnn)
        self.add_module('bn', bn)
        self.add_module('fc_phase', fc_phase)
        self.add_module('chimera', chimera_net)

    def forward(self, input):
        """
        input:
            x_mag: B X T X F X C tensor,
                   T is frame_length
                   F is frequency bin
                   C is num_speaker
            x_phase: B X T X F X (2C) tensor
        """
        assert len(input) == 2, "There must be 2 tensors in the input for phase network"
        [x_mag, x_phase] = input
        [embedding, mask_A, mask_B] = self.chimera([x_mag])
        batch_size, frame_size, frequency_size = mask_A.size()
        mag_A = x_mag * mask_A
        mag_B = x_mag * mask_B
        input_A = torch.cat((mag_A, x_phase.view(batch_size, frame_size,-1)), 2)
        rnn_output_A, _ = self.rnn(input_A)
        rnn_output_A = rnn_output_A.permute(0, 2, 1)
        rnn_output_A = self.bn(rnn_output_A)
        rnn_output_A = rnn_output_A.permute(0, 2, 1)
        phase_A  = self.fc_phase(rnn_output_A)
        phase_A = phase_A.reshape(batch_size, frame_size, frequency_size, -1)
        input_B = torch.cat((mag_B, x_phase.view(batch_size, frame_size,-1)), 2)
        rnn_output_B, hidden = self.rnn(input_B)
        rnn_output_B = rnn_output_B.permute(0, 2, 1)
        rnn_output_B = self.bn(rnn_output_B)
        rnn_output_B = rnn_output_B.permute(0, 2, 1)
        phase_B = self.fc_phase(rnn_output_B)
        phase_B = phase_B.reshape(batch_size, frame_size, frequency_size, -1)
        phase_A = phase_A + x_phase
        phase_B = phase_B + x_phase
        phase_A = F.normalize(phase_A, p=2, dim=-1)
        phase_B = F.normalize(phase_B, p=2, dim=-1)
        return [embedding, x_mag, mask_A, mask_B, phase_A, phase_B]
