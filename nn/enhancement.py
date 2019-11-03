import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import LayerNormLSTM

class enhance(nn.Module):
    """
    The model predicts the clean magnitude from a noisy magnitude.
    To reduce the mismatch between the noisy and clean signal,
    "restoration" layers (https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0515.PDF)
    are appliled after the noisy magnitude and before the spectrogram estimation layer.
    """
    def __init__(self, model_options):
        super(enhance, self).__init__()
        self.input_dim = model_options.input_dim
        self.output_dim = model_options.output_dim if "output_dim" in model_options else self.input_dim
        self.hidden_dim = model_options.hidden_dim if "hidden_dim" in model_options else 300
        self.num_layers = model_options.num_layers if "num_layers" in model_options else 3
        self.dropout = model_options.dropout if "dropout" in model_options else 0.3
        rnn = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first = True
        )
        bn = nn.BatchNorm1d(self.hidden_dim*2)
        fc_mi = nn.Linear(self.hidden_dim * 2, self.output_dim)
        fc_pre = nn.Linear(self.output_dim, self.output_dim)
        fc_post = nn.Linear(self.output_dim, self.output_dim)
        self.add_module('rnn', rnn)
        self.add_module('bn', bn)
        self.add_module('fc_mi', fc_mi)
        self.add_module('fc_pre', fc_pre)
        self.add_module('fc_post', fc_post)

    def forward(self, input):
        # x is N*T*F tensor
        x, mag_noisy = input
        batch_size, frame_size, _ = x.size()
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
