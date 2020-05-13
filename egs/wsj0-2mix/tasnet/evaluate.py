import sys
sys.path.append('../../../../../onssen/')

from onssen import utils
from sklearn.cluster import KMeans
import librosa
import numpy as np
import torch


class tester_tasnet(utils.tester):
    def get_est_sig(self, input, label, output):
        """
        args:
            feature_mix: batch x frame x frequency
            embedding: batch x frame x frequency x embedding_dim
            stft_r_mix: batch x frame x frequency
            stft_i_mix: batch x frame x frequency
            sig_ref: batch x num_spk x nsample
        return:
            sig_est: batch x num_spk x nsample
        """
        feature_mix, = input
        sig_ref, = label
        batch, num_spk, N = sig_ref.shape
        sig_est = torch.zeros((batch, num_spk, N), device=self.device)
        for i in range(num_spk):
            sig_est[:, i, :] = output[i][0:N]
        return sig_est, sig_ref

