"""
We need to define a batch size for training the deep clustering model.
Each batch has a shape (batch_size, 100/400, feature_dim)

For STFT:
8kHz fs
32 ms window length 32*8 = 256
8 ms window shift = 64

44kHz fs
128 ms window length 128*8 = 1024
32 ms window shift = 256
"""

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from .feature_utils import *
import glob
import numpy as np
import random
import torch
import torchaudio
import torch.nn.functional as F


def wsj0_2mix_dataloader(model_name, feature_options, partition, device=None):
    if partition == "tr" or partition == "cv":
        return DataLoader(
            wsj0_2mix_dataset(model_name, feature_options, partition, device=device),
            batch_size=feature_options.batch_size,
            shuffle=True,
        )
    elif partition == "tt":
        return DataLoader(
            wsj0_2mix_eval_dataset(model_name, feature_options, partition, device=device),
            batch_size=1,
        )


class wsj0_2mix_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, device=None):
        """
        The arguments:
            feature_options: a dictionary containing the feature params
            partition: can be "tr", "cv"
            model_name: can be "dc", "chimera", "chimera++", "phase"
            e.g.
            "feature_options": {
                "data_path": "/home/data/wsj0-2mix",
                "batch_size": 16,
                "frame_length": 400,
                "sampling_rate": 8000,
                "window_size": 256,
                "hop_size": 64,
                "db_threshold": 40
            }
        The returns:
            input: a tuple which follows the requirement of the loss
            label: a tuple which follows the requirement of the loss
            e.g.
            for dc loss:
                input: (feature_mix)
                label: (one_hot_label)
            for chimera loss:
                input: (feature_mix)
                label: (one_hot_label, mag_mix, mag_s1, mag_s2)
        """
        self.model_name = model_name
        self.sampling_rate = feature_options.sampling_rate
        if self.model_name in ["lstm-tasnet", "conv-tasnet"]:
            self.chunk_size = feature_options.chunk_size
        else:
            self.window_size = feature_options.window_size
            self.hop_size = feature_options.hop_size
            self.frame_length = feature_options.frame_length
            self.db_threshold = feature_options.db_threshold
        self.file_list = []
        full_path = feature_options.data_path+'/wav8k/min/'+partition+'/mix/*.wav'
        self.file_list = glob.glob(full_path)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device


    def get_tr_sigs(self, fn, sr):
        sig, rate = torchaudio.load(fn)
        assert(rate==sr)
        sig_s1, rate = torchaudio.load(fn.replace('/mix','/s1'))
        sig_s2, rate = torchaudio.load(fn.replace('/mix','/s2'))
        if sig.shape[1] < self.chunk_size:
            gap = self.chunk_size- sig.shape[1]
            sig = F.pad(sig, (0, gap), mode='constant')
            sig_s1 = F.pad(sig_s1, (0, gap), mode='constant')
            sig_s2 = F.pad(sig_s2, (0, gap), mode='constant')
        else:
            random_start = random.randint(0, sig.shape[1]-self.chunk_size)
            sig = sig[:, random_start:self.chunk_size+random_start]
            sig_s1 = sig_s1[:, random_start:self.chunk_size+random_start]
            sig_s2 = sig_s2[:, random_start:self.chunk_size+random_start]
        return sig, sig_s1, sig_s2

    def get_feature(self,fn):
        if self.model_name in ["lstm-tasnet", "conv-tasnet"]:
            sig_mix, sig_s1, sig_s2 = self.get_tr_sigs(fn, self.sampling_rate)
            sig_mix = sig_mix.reshape(-1,)
            sig_s1 = sig_s1.reshape(-1,)
            sig_s2 = sig_s2.reshape(-1,)
            input, label = [sig_mix], [sig_s1, sig_s2]
            input = [ele.to(self.device) for ele in input]
            label = [ele.to(self.device) for ele in label]
            return input, label

        stft_mix = get_stft(fn, self.sampling_rate, self.window_size, self.hop_size)
        stft_s1 = get_stft(fn.replace('/mix','/s1'), self.sampling_rate, self.window_size, self.hop_size)
        stft_s2 = get_stft(fn.replace('/mix','/s2'), self.sampling_rate, self.window_size, self.hop_size)

        if stft_mix.shape[0]<=self.frame_length:
            #pad in a double-copy fashion
            times = self.frame_length // stft_mix.shape[0]+1
            stft_mix = np.concatenate([stft_mix]*times, axis=0)
            stft_s1 = np.concatenate([stft_s1]*times, axis=0)
            stft_s2 = np.concatenate([stft_s2]*times, axis=0)

        random_index = np.random.randint(stft_mix.shape[0]-self.frame_length)
        stft_mix = stft_mix[random_index:random_index+self.frame_length]
        stft_s1 = stft_s1[random_index:random_index+self.frame_length]
        stft_s2 = stft_s2[random_index:random_index+self.frame_length]
        # base feature
        feature_mix = get_log_magnitude(stft_mix)
        # one_hot_label
        mag_mix = np.abs(stft_mix)
        mag_s1 = np.abs(stft_s1)
        mag_s2 = np.abs(stft_s2)
        one_hot_label = get_one_hot(feature_mix, mag_s1, mag_s2, self.db_threshold)

        if self.model_name == "dc":
            input, label = [feature_mix], [one_hot_label, mag_mix]

        if self.model_name == "chimera":
            input, label = [feature_mix], [one_hot_label, mag_mix, mag_s1, mag_s2]

        if self.model_name == "chimera++":
            cos_s1 = get_cos_difference(stft_mix, stft_s1)
            cos_s2 = get_cos_difference(stft_mix, stft_s2)
            input, label = [feature_mix], [one_hot_label, mag_mix, mag_s1, mag_s2, cos_s1, cos_s2]

        if self.model_name == "phase":
            phase_mix = get_phase(stft_mix)
            phase_s1 = get_phase(stft_s1)
            phase_s2 = get_phase(stft_s2)
            input, label = [feature_mix, phase_mix], [one_hot_label, mag_mix, mag_s1, mag_s2, phase_s1, phase_s2]

        
        input = [torch.tensor(ele).to(self.device) for ele in input]
        label = [torch.tensor(ele).to(self.device) for ele in label]

        return input, label


    def __getitem__(self, index):
        file_name_mix = self.file_list[index]
        return self.get_feature(file_name_mix)


    def __len__(self):
        return len(self.file_list)


class wsj0_2mix_eval_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, device=None):
        """
        The arguments:
            feature_options: a dictionary containing the feature params
            partition: can be "tr", "cv"
            model_name: can be "dc", "chimera", "chimera++", "phase"
            e.g.
            "feature_options": {
                "data_path": "/home/data/wsj0-2mix",
                "batch_size": 16,
                "frame_length": 400,
                "sampling_rate": 8000,
                "window_size": 256,
                "hop_size": 64,
                "db_threshold": 40
            }
        The returns:
            input: a tuple which follows the requirement of the loss
            label: a tuple which follows the requirement of the loss
            e.g.
            for dc loss:
                input: (feature_mix)
                label: (one_hot_label)
            for chimera loss:
                input: (feature_mix)
                label: (one_hot_label, mag_mix, mag_s1, mag_s2)
        """
        self.model_name = model_name
        self.sampling_rate = feature_options.sampling_rate
        if self.model_name in ["lstm-tasnet", "conv-tasnet"]:
            self.chunk_size = feature_options.chunk_size
        else:
            self.window_size = feature_options.window_size
            self.hop_size = feature_options.hop_size
            self.frame_length = feature_options.frame_length
            self.db_threshold = feature_options.db_threshold
        self.file_list = []
        full_path = feature_options.data_path+'/wav8k/min/'+partition+'/mix/*.wav'
        self.file_list = glob.glob(full_path)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device


    def get_sigs(self, fn, sr):
        sig_mix, rate = torchaudio.load(fn)
        assert(rate==sr)
        sig_s1, rate = torchaudio.load(fn.replace('tt/mix/','tt/s1/'))
        sig_s2, rate = torchaudio.load(fn.replace('tt/mix/','tt/s2/'))
        N = sig_mix.shape[1]
        gap = 32- N % 32
        sig_mix = F.pad(sig_mix, (0, gap), mode='constant')
        sig_s1 = F.pad(sig_s1, (0, gap), mode='constant')
        sig_s2 = F.pad(sig_s2, (0, gap), mode='constant')
        sig_ref = torch.cat((sig_s1, sig_s2), dim=0)
        sig_mix = sig_mix.reshape(-1,)
        return sig_mix, sig_ref


    def get_feature(self,fn):
        if self.model_name in ["lstm-tasnet", "conv-tasnet"]:
            sig_mix, sig_ref = self.get_sigs(fn, self.sampling_rate)
            input, label = [sig_mix.to(self.device)], [sig_ref.to(self.device)]
        else:
            stft_mix = get_stft(fn, self.sampling_rate, self.window_size, self.hop_size)
            stft_r_mix = np.real(stft_mix)
            stft_i_mix = np.imag(stft_mix)
            feature_mix = get_log_magnitude(stft_mix)
            sig_ref = self.get_ref_sig(fn)
            input, label = [feature_mix], [stft_r_mix, stft_i_mix, sig_ref]
            input = [torch.tensor(ele).to(self.device) for ele in input]
            label = [torch.tensor(ele).to(self.device) for ele in label]

        return input, label


    def __getitem__(self, index):
        file_name_mix = self.file_list[index]
        return self.get_feature(file_name_mix)


    def __len__(self):
        return len(self.file_list)
