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
import glob,librosa, numpy as np, os, random, torch

"""
what I want from the DataLoader?
Each batch should contain different speakers
There should be 400 * 513 tensor for each sample
We have a list of files, each contains N times of 400 frames
To make full use of them, we need to generate something
We have 20 speakers, each will contains N X 400 X 513 tensors

"""

def daps_enhance_dataloader(num_batch, feature_options, partition, device=None):
        return DataLoader(
            daps_dataset(num_batch, feature_options, partition, device=device),
            batch_size=feature_options.batch_size,
            shuffle=True,
        )


class daps_dataset(Dataset):
    def __init__(self, num_batch, feature_options, partition, device=None):
        """
        The arguments:
            feature_options: a dictionary containing the feature params
            partition: can be "train", "validation"
            num_batch: Each training epoch uses num_batch * batch_size * frame_length data
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
        self.sampling_rate = feature_options.sampling_rate
        self.window_size = feature_options.window_size
        self.hop_size = feature_options.hop_size
        self.frame_length = feature_options.frame_length
        self.num_batch = num_batch
        self.batch_size = feature_options.batch_size
        self.file_list = []
        self.base_path = feature_options.data_path
        self.partition = partition
        self.length_remaining = 0
        self.get_item_list()
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device


    def get_item_list(self):
        f = open(self.base_path+'/'+self.partition)
        self.file_list = [line.replace('\n','') for line in f]
        random.shuffle(self.file_list)


    def __getitem__(self, index):
        if self.length_remaining < self.frame_length:
            if len(self.file_list)==0:
                self.get_item_list()
            # add one more file, delete the index from the list
            index = index % len(self.file_list)
            f_noisy = self.file_list.pop(index)
            base_names = os.path.basename(f_noisy).split("_")
            f_clean = self.base_path + "/clean/" + base_names[0] + "_" + base_names[1] + "_clean.wav"
            stft_noisy = get_stft(f_noisy, self.sampling_rate, self.window_size, self.hop_size)
            stft_clean = get_stft(f_clean, self.sampling_rate, self.window_size, self.hop_size)

            feature = get_log_magnitude(stft_noisy)
            #feature = get_log_mel_spectrogram(f_noisy, self.sampling_rate, self.window_size, self.hop_size)
            # one_hot_label
            mag_noisy = np.abs(stft_noisy)
            mag_clean = np.abs(stft_clean)
            cos_diff = get_cos_difference(stft_noisy, stft_clean)
            input, label = [feature, mag_noisy], [mag_clean, cos_diff]

            input = [torch.tensor(ele).to(self.device) for ele in input]
            label = [torch.tensor(ele).to(self.device) for ele in label]

            self.input = input
            self.label = label
            return self.cutoff_feature()
        else:
            return self.cutoff_feature()

    def cutoff_feature(self):
        input, label =  [ele[0:self.frame_length] for ele in self.input], [ele[0:self.frame_length] for ele in self.label]
        self.input = [ele[self.frame_length:] for ele in self.input]
        self.label = [ele[self.frame_length:] for ele in self.label]
        self.length_remaining = self.input[0].shape[0]
        return input, label

    def __len__(self):
        return self.num_batch*self.batch_size
