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
import glob
import librosa
import numpy as np


def wsj0_2mix_dataloader(model_name, feature_options, partition, cuda):
        return DataLoader(
            WSJ0_2Mix_Dataset(model_name, feature_options, partition, cuda),
            batch_size=feature_options.batch_size,
            shuffle=True,
        )


class WSJ0_2Mix_Dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda):
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
        self.sampling_rate = feature_options.sampling_rate
        self.window_size = feature_options.window_size
        self.hop_size = feature_options.hop_size
        self.frame_length = feature_options.frame_length
        self.db_threshold = feature_options.db_threshold
        self.model_name = model_name
        self.cuda = cuda
        self.file_list = []
        full_path = feature_options.data_path+'/wav8k/min/'+partition+'/mix/*.wav'
        self.file_list = glob.glob(full_path)


    def get_phase(self, stft):
        real = np.real(stft)
        imag = np.imag(stft)
        phase = np.array([real, imag])
        phase = np.transpose(phase, (1,2,0))
        return phase


    def get_angle(self, stft):
        angle = (np.angle(stft)%(2*np.pi)+2*np.pi) %(2*np.pi)
        return angle


    def get_one_hot(self,feature_mix, mag_s1, mag_s2):
        tf1 = np.log10(mag_s1+1e-14)
        tf2 = np.log10(mag_s2+1e-14)
        specs = np.asarray([tf1, tf2])
        vals = np.argmax(specs, axis=0)
        Y = np.zeros(tf1.shape+(2,))
        for i in range(2):
            temp = np.zeros((2))
            temp[i]=1
            Y[vals == i] = temp
        #label the silence part
        m = np.max(feature_mix) - self.db_threshold/20
        temp = np.zeros((2))
        Y[feature_mix < m] = temp
        return Y


    def get_feature(self,fn):
        sig_mix, fs = librosa.load(fn, sr=self.sampling_rate)
        sig_s1, fs = librosa.load(fn.replace('/mix','/s1'), sr=self.sampling_rate)
        sig_s2, fs = librosa.load(fn.replace('/mix','/s2'), sr=self.sampling_rate)
        stft_mix = np.transpose(librosa.core.stft(sig_mix, n_fft=self.window_size, hop_length=self.hop_size))
        stft_s1 = np.transpose(librosa.core.stft(sig_s1, n_fft=self.window_size, hop_length=self.hop_size))
        stft_s2 = np.transpose(librosa.core.stft(sig_s2, n_fft=self.window_size, hop_length=self.hop_size))
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
        feature_mix = np.log10(np.abs(stft_mix)+1e-7)
        # one_hot_label
        mag_mix = np.abs(stft_mix)
        mag_s1 = np.abs(stft_s1)
        mag_s2 = np.abs(stft_s2)
        one_hot_label = self.get_one_hot(feature_mix, mag_s1, mag_s2)

        if self.model_name == "dc":
            input, label = [feature_mix], [one_hot_label]

        if self.model_name == "chimera":
            input, label = [feature_mix], [one_hot_label, mag_mix, mag_s1, mag_s2]

        if self.model_name == "chimera++":
            phase_mix = self.get_angle(stft_mix)
            phase_s1 = self.get_angle(stft_s1)
            phase_s2 =self. get_angle(stft_s2)
            cos_s1 = np.cos(phase_mix - phase_s1)
            cos_s2 = np.cos(phase_mix - phase_s2)
            input, label = [feature_mix], [one_hot_label, mag_mix, mag_s1, mag_s2, cos_s1, cos_s2]

        if self.model_name == "phase":
            phase_mix = self.get_phase(stft_mix)
            phase_s1 = self.get_phase(stft_s1)
            phase_s2 = self.get_phase(stft_s2)
            input, label = [feature_mix, phase_mix], [one_hot_label, mag_mix, mag_s1, mag_s2, phase_s1, phase_s2]

        if self.cuda == "True":
            input = [torch.Tensor(ele).cuda() for ele in input]
            label = [torch.Tensor(ele).cuda() for ele in label]

        return input, label


    def __getitem__(self, index):
        file_name_mix = self.file_list[index]
        return self.get_feature(file_name_mix)


    def __len__(self):
        return len(self.file_list)
