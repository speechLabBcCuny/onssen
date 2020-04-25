import torch
import torch.nn as nn


class enhance(nn.Module):
    """
    The model predicts the clean magnitude from a noisy magnitude.
    To reduce the mismatch between the noisy and clean signal,
    "restoration" layers (https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0515.PDF)
    are appliled after the noisy magnitude and before the spectrogram estimation layer.
    """
    def __init__(
        self, 
        input_dim,
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
        bn = nn.BatchNorm1d(hidden_dim*2)
        fc_mi = nn.Linear(hidden_dim * 2, input_dim)
        fc_pre = nn.Linear(input_dim, input_dim)
        fc_post = nn.Linear(input_dim, input_dim)
        self.add_module('rnn', rnn)
        self.add_module('bn', bn)
        self.add_module('fc_mi', fc_mi)
        self.add_module('fc_pre', fc_pre)
        self.add_module('fc_post', fc_post)

    def forward(self, input):
        assert len(input)==2, "There must be two tensors in the input for the enhance network"
        # x is N*T*F tensor
        x, mag_noisy = input
        batch_size, frame, frequency = x.size()
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(x)
        rnn_output = rnn_output.permute(0, 2, 1)
        rnn_output = self.bn(rnn_output)
        rnn_output = rnn_output.permute(0, 2, 1)
        mask = torch.sigmoid(self.fc_mi(rnn_output))
        mag_noisy_re = torch.relu(self.fc_pre(mag_noisy))
        clean_est = torch.mul(mag_noisy_re, mask)
        clean = torch.relu(self.fc_post(clean_est))
        return [clean]
