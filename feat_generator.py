'''
We need to define a batch size for training the deep clustering model.
Each batch has a shape (batch_size, 100/400, 512)
Each time push 100 wav files to a list, generate ? * 100 * 512 features
When the left size is less than batch_size, recall generate_samples method

For STFT:
8kHz fs
32 ms window length 32*8 = 256
8 ms window shift = 64
square root of the hann window
'''
from config import *
import scipy
import numpy as np
import librosa
from stft_utils import e_stft, e_istft

def normalize_feat(fn):
    sig, fs = librosa.load(fn,sr=SAMPLING_RATE)
    assert(fs==SAMPLING_RATE)
    # sig = sig-np.mean(sig)
    # sig = sig/(np.max(np.abs(sig))) +1e-7
    n_sample = sig.shape[0]
    stft = e_stft(sig,WINDOW_LENGTH,HOG,'hamming')
    abs_tf = np.log10(np.abs(np.transpose(stft))+1e-7)
    remain = abs_tf.shape[0]%100
    #abs_tf = np.concatenate((abs_tf,abs_tf[:remain,:]),axis = 0)
    return abs_tf[:-remain,:]

def get_stft(fn):
    sig, fs = librosa.load(fn,sr=SAMPLING_RATE)
    assert(fs==SAMPLING_RATE)
    stft = e_stft(sig,WINDOW_LENGTH,HOG,'hamming')
    stft = np.log10(np.abs(np.transpose(stft))+1e-7)
    remain = stft.shape[0]%100
    #stft = np.concatenate((stft,stft[:remain,:]),axis = 0)
    return stft[:-remain,:]

def get_magnitude(magnitude, fn):
    sig, fs = librosa.load(fn,sr=SAMPLING_RATE)
    n_sample = sig.shape[0]
    stft = e_stft(sig,WINDOW_LENGTH,HOG,'hamming')
    abs_tf = np.abs(np.transpose(stft))
    remain = abs_tf.shape[0]%100
    abs_tf = abs_tf[:-remain,:]
    #abs_tf = np.concatenate((abs_tf,abs_tf[:remain,:]),axis = 0)
    if magnitude is not None:
        magnitude = np.concatenate((magnitude,abs_tf),axis = 0)
    else:
        magnitude = abs_tf
    return magnitude

def get_feature(feat, fn):
    tf= normalize_feat(fn)
    if feat is not None:
        feat = np.concatenate((feat,tf),axis = 0)
    else:
        feat = tf
    return feat

def get_one_hot(target, fn):
    tf_mix = get_stft(fn)
    if tf_mix.shape[0]==0:
        return target
    fn_s1 = fn.replace('mix','s1')
    fn_s2 = fn.replace('mix','s2')
    tf1 = get_stft(fn_s1)
    tf2 = get_stft(fn_s2)
    #do we need normalize the spectrogram?
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
#     temp[2]=1
    Y[tf_mix < m] = temp
    if target is None:
        return Y
    else:
        return np.concatenate((target,Y),axis = 0)


def generate_samples_chimera_net(f_list, batch_size=32):
    #generate X * 100 * 129 feature
    # and     X * 100 * 3   label
    magnitude = None
    feat = None
    target = None
    while (feat is None or feat.shape[0]<100*batch_size) and len(f_list)>0:
        #feature part
        fn = f_list.pop(0)
        f_list.append(fn)
        magnitude = get_magnitude(magnitude, fn)
        feat = get_feature(feat, fn)
        target = get_one_hot(target, fn)
    if feat.shape[0]<batch_size*100:
        return (None,None,None)
    feat = feat[0:batch_size*100,:]
    target = target[0:batch_size*100,:]
    magnitude = magnitude.reshape((-1,100,129))
    feat = feat.reshape((-1,100,129))
    target = target.reshape((-1,100,129,NUM_SPEAKER))
    return (magnitude, feat, target)
