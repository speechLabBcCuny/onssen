from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from .feature_utils import *
import glob
import numpy as np
import random
import torch


def edinburgh_tts_dataloader(model_name, feature_options, partition, device=None):
        return DataLoader(
            edinburgh_tts_dataset(model_name, feature_options, partition, device=device),
            batch_size=feature_options.batch_size,
            shuffle=True,
        )


class edinburgh_tts_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, device=None):
        """
        The arguments:
            feature_options: a dictionary containing the feature params
            partition: can be "train", "validation"
            model_name: can be "dc", "chimera", "chimera++", "phase"
            e.g.
            "feature_options": {
                "data_path": "/home/data/Edinburg_tts",
                "batch_size": 16,
                "frame_length": 400,
                "sampling_rate": 16000,
                "window_size": 512,
                "hop_size": 128,
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
        self.sampling_rate = feature_options.sampling_rate
        self.window_size = feature_options.window_size
        self.hop_size = feature_options.hop_size
        self.frame_length = feature_options.frame_length
        self.db_threshold = feature_options.db_threshold
        self.model_name = model_name
        self.data_path = feature_options.data_path
        self.partition = partition
        self.file_list = []
        self.get_file_list()
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def get_file_list(self):
        with open(self.data_path+'/'+self.partition,'r') as f:
            for line in f:
                self.file_list.append(self.data_path+'/noisy_trainset_28spk_wav/'+line.replace('\n',''))
        random.shuffle(self.file_list)


    def get_feature(self,fn):
        stft_mix = get_stft(fn, self.sampling_rate, self.window_size, self.hop_size)
        stft_s1 = get_stft(fn.replace('/noisy_trainset_28spk_wav','/clean_trainset_28spk_wav'), self.sampling_rate, self.window_size, self.hop_size)
        stft_s2 = get_stft_from_subtraction(fn, fn.replace('/noisy_trainset_28spk_wav','/clean_trainset_28spk_wav'), self.sampling_rate, self.window_size, self.hop_size)

        if stft_mix.shape[0]<=self.frame_length:
            #pad in a double-copy fashion
            times = self.frame_length // stft_mix.shape[0]+1
            stft_mix = np.concatenate([stft_mix]*times, axis=0)
            stft_s1 = np.concatenate([stft_s1]*times, axis=0)
            stft_s2 = np.concatenate([stft_s2]*times, axis=0)

        stft_mix = stft_mix[:self.frame_length]
        stft_s1 = stft_s1[:self.frame_length]
        stft_s2 = stft_s2[:self.frame_length]
        # base feature
        feature_mix = get_log_magnitude(stft_mix)
        # one_hot_label
        mag_mix = np.abs(stft_mix)
        mag_s1 = np.abs(stft_s1)
        mag_s2 = np.abs(stft_s2)
        one_hot_label = get_one_hot(feature_mix, mag_s1, mag_s2, self.db_threshold)

        if self.model_name == "dc":
            input, label = [feature_mix], [one_hot_label]

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
