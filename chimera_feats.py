from config import *
import glob
import librosa
import numpy as np
import os
import random
from stft_utils import e_stft, e_istft

def get_log_stft(stft):
    return np.log10(np.abs(stft.T)+1e-7)

def get_mag_angle(stft):
    stft = stft.T
    angle = np.angle(stft)
    magnitude = np.abs(stft)
    mag_angle = np.array([magnitude, angle])
    return mag_angle

def get_one_hot(stft_mix, stft_s1, stft_s2):
    tf1 = np.log10(np.abs(stft_s1.T)+1e-7)
    tf2 = np.log10(np.abs(stft_s2.T)+1e-7)
    tf_mix = np.log10(np.abs(stft_mix.T)+1e-7)
    specs = np.asarray([tf1, tf2])
    vals = np.argmax(specs, axis=0)
    Y = np.zeros(tf1.shape+(NUM_SPEAKER,))
    for i in range(2):
        temp = np.zeros((NUM_SPEAKER))
        temp[i]=1
        Y[vals == i] = temp
    #label the silence part
    m = np.max(tf_mix) - DB_THRESHOLD/20
    temp = np.zeros((NUM_SPEAKER))
    Y[tf_mix < m] = temp
    return Y

def get_speaker_dict(path):
    speakers = {}
    SAMPLING_RATE=8000
    f_list = glob.glob(path)
    for ele in f_list:
        s1 = os.path.basename(ele).split('_')[0][:3]
        s2 = os.path.basename(ele).split('_')[2][:3]
        if s1 not in speakers:
            speakers[s1] = [ele.replace('mix','s1')]
        else:
            speakers[s1].append(ele.replace('mix','s1'))
        if s2 not in speakers:
            speakers[s2] = [ele.replace('mix','s2')]
        else:
            speakers[s1].append(ele.replace('mix','s2'))
    return speakers

def get_chimera_samples(speakers):
    batch_count = 0
    batch_x = []
    batch_y = []
    batch_mag_angle_mix = []
    batch_mag_angle_s1 = []
    batch_mag_angle_s2 = []
    while True:
        s1, s2 = random.sample(list(speakers.keys()),2)
        fn_s1 = random.choice(speakers[s1])
        fn_s2 = random.choice(speakers[s2])
        sig_s1, fs = librosa.load(fn_s1,sr=SAMPLING_RATE)
        sig_s2, fs = librosa.load(fn_s2,sr=SAMPLING_RATE)
        sig_s1 = sig_s1[0:min(sig_s1.shape[0],sig_s2.shape[0])]
        sig_s2 = sig_s2[0:min(sig_s1.shape[0],sig_s2.shape[0])]
        sig_mix = sig_s1+sig_s2
        stft_s1 = e_stft(sig_s1,WINDOW_LENGTH,HOG,'hann')
        stft_s2 = e_stft(sig_s2,WINDOW_LENGTH,HOG,'hann')
        stft_mix = e_stft(sig_mix,WINDOW_LENGTH,HOG,'hann')
        feature = get_log_stft(stft_mix)
        one_hot_mask = get_one_hot(stft_mix, stft_s1, stft_s2)
        mag_angle_s1 = get_mag_angle(stft_s1)
        mag_angle_s2 = get_mag_angle(stft_s2)
        mag_angle_mix = get_mag_angle(stft_mix)
        i = 0
        while i + FRAME_LENGTH < feature.shape[0]:
            batch_x.append(feature[i:i+FRAME_LENGTH])
            batch_y.append(one_hot_mask[i:i+FRAME_LENGTH])
            batch_mag_angle_mix.append(mag_angle_mix[:,i:i+FRAME_LENGTH])
            batch_mag_angle_s1.append(mag_angle_s1[:,i:i+FRAME_LENGTH])
            batch_mag_angle_s2.append(mag_angle_s2[:,i:i+FRAME_LENGTH])
            i += FRAME_LENGTH//2
            batch_count = batch_count+1
            if batch_count == BATCH_SIZE:
                batch_x = np.array(batch_x).reshape((BATCH_SIZE, FRAME_LENGTH, FREQUENCY_SIZE))
                batch_y = np.array(batch_y).reshape((BATCH_SIZE, FRAME_LENGTH, FREQUENCY_SIZE, NUM_SPEAKER))
                batch_mag_angle_mix = np.array(batch_mag_angle_mix).reshape((-1,BATCH_SIZE, FRAME_LENGTH, FREQUENCY_SIZE))
                batch_mag_angle_s1 = np.array(batch_mag_angle_s1).reshape((-1,BATCH_SIZE, FRAME_LENGTH, FREQUENCY_SIZE))
                batch_mag_angle_s2 = np.array(batch_mag_angle_s2).reshape((-1,BATCH_SIZE, FRAME_LENGTH, FREQUENCY_SIZE))
                return (batch_mag_angle_mix, batch_mag_angle_s1, batch_mag_angle_s2, batch_x, batch_y)
