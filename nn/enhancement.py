import torch
import torch.nn as nn
import torch.nn.functional as F

class enhance(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=300,
        num_layers=3,
        dropout=0.3
    ):
        super(enhance, self).__init__()
        rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first = True
        )
        fc_mi = nn.Linear(hidden_dim * 2, output_dim)
        fc_pre = nn.Linear(output_dim, output_dim)
        fc_post = nn.Linear(output_dim, output_dim)
        self.output_dim = output_dim
        self.add_module('rnn', rnn)
        self.add_module('fc_mi', fc_mi)
        self.add_module('fc_pre', fc_pre)
        self.add_module('fc_post', fc_post)

    def forward(self, input):
        # x is N*T*F tensor
        x, mag_noisy = input
        batch_size, frame_size, _ = x.size()
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(x)
        mask = torch.sigmoid(self.fc_mi(rnn_output))
        mag_noisy_re = torch.relu(self.fc_pre(mag_noisy))
        clean_est = torch.mul(mag_noisy_re, mask)
        clean = torch.relu(self.fc_post(clean_est))
        return [clean]
