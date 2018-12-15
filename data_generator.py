from torch.utils.data.dataset import Dataset
import glob
import librosa
import numpy as np
from stft_utils import e_stft, e_istft

class WSJ0_2Mix_Dataset(Dataset):
    def __init__(self, mode):
        # mode can be tr, cv, and test
        self.file_list = []
        f_list = glob.glob('/scratch/near/2speakers/wav8k/min/'+mode+'/mix/*.wav')
        for f_name in f_list:
            sig_mix, fs = librosa.load(f_name,sr=8000)
            if (sig_mix.shape[0]-192)//64>=402:
                self.file_list.append(f_name)

    def get_phase(self, stft):
        real = np.real(stft)
        imag = np.imag(stft)
        phase = np.array([real, imag])
        phase = np.transpose(phase, (1,2,0))
        return phase

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
        m = np.max(feature_mix) - 40/20
        temp = np.zeros((2))
        Y[feature_mix < m] = temp
        return Y

    def get_feature(self,fn):
        sig_mix, fs = librosa.load(fn,sr=8000)
        sig_s1, fs = librosa.load(fn.replace('mix','s1'), sr=8000)
        sig_s2, fs = librosa.load(fn.replace('mix','s2'), sr=8000)
        stft_mix = np.transpose(e_stft(sig_mix,256,64,'hann'))
        stft_s1 = np.transpose(e_stft(sig_s1,256,64,'hann'))
        stft_s2 = np.transpose(e_stft(sig_s2,256,64,'hann'))
        if stft_mix.shape[0]<400:
            return None
        random_index = np.random.randint(stft_mix.shape[0]-400)
        stft_mix = stft_mix[random_index:random_index+400]
        stft_s1 = stft_s1[random_index:random_index+400]
        stft_s2 = stft_s2[random_index:random_index+400]
        # input feature
        feature_mix = np.log10(np.abs(stft_mix)+1e-14)
        # magnitude
        mag_mix = np.abs(stft_mix)
        mag_s1 = np.abs(stft_s1)
        mag_s2 = np.abs(stft_s2)
        # target
        target = self.get_one_hot(feature_mix, mag_s1, mag_s2)
        # phase
        phase_mix = self.get_phase(stft_mix)
        phase_s1 = self.get_phase(stft_s1)
        phase_s2 =self. get_phase(stft_s2)

        return feature_mix, target, mag_mix, phase_mix, mag_s1, mag_s2, phase_s1, phase_s2

    def __getitem__(self, index):
        file_name_mix = self.file_list[index]
        return self.get_feature(file_name_mix)

    def __len__(self):
        return len(self.file_list)
