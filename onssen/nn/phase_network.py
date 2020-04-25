import torch
import torch.nn as nn
import torch.nn.functional as F
from .chimera import chimera


class phase_net(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim=300,
        num_layers=3,
        embedding_dim=20,
        dropout=0.3,
        num_speaker=2,
    ):
        super(phase_net, self).__init__()
        chimera_net = chimera(input_dim, hidden_dim, num_layers, embedding_dim, dropout, num_speaker)
        rnn = nn.LSTM(
            input_dim*3,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first = True
        )
        bn = nn.BatchNorm1d(hidden_dim*2)
        fc_phase = nn.Linear(hidden_dim*2, num_speaker*output_dim)
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
        batch_size, frame, frequency = mask_A.size()
        mag_A = x_mag * mask_A
        mag_B = x_mag * mask_B
        input_A = torch.cat((mag_A, x_phase.view(batch_size, frame,-1)), 2)
        rnn_output_A, _ = self.rnn(input_A)
        rnn_output_A = rnn_output_A.permute(0, 2, 1)
        rnn_output_A = self.bn(rnn_output_A)
        rnn_output_A = rnn_output_A.permute(0, 2, 1)
        phase_A  = self.fc_phase(rnn_output_A)
        phase_A = phase_A.reshape(batch_size, frame, frequency, -1)
        input_B = torch.cat((mag_B, x_phase.view(batch_size, frame,-1)), 2)
        rnn_output_B, hidden = self.rnn(input_B)
        rnn_output_B = rnn_output_B.permute(0, 2, 1)
        rnn_output_B = self.bn(rnn_output_B)
        rnn_output_B = rnn_output_B.permute(0, 2, 1)
        phase_B = self.fc_phase(rnn_output_B)
        phase_B = phase_B.reshape(batch_size, frame, frequency, -1)
        phase_A = phase_A + x_phase
        phase_B = phase_B + x_phase
        phase_A = F.normalize(phase_A, p=2, dim=-1)
        phase_B = F.normalize(phase_B, p=2, dim=-1)
        return [embedding, mask_A, mask_B, phase_A, phase_B]
