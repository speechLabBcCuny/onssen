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

def normalize_feat(fn, num_frame):
    sig, fs = librosa.load(fn,sr=SAMPLING_RATE)
    assert(fs==SAMPLING_RATE)
    # sig = sig-np.mean(sig)
    # sig = sig/(np.max(np.abs(sig))) +1e-7
    n_sample = sig.shape[0]
    stft = e_stft(sig,WINDOW_LENGTH,HOG,'hann')
    abs_tf = np.log10(np.abs(np.transpose(stft))+1e-7)
    remain = abs_tf.shape[0]%num_frame
    #abs_tf = np.concatenate((abs_tf,abs_tf[:remain,:]),axis = 0)
    return abs_tf[:-remain,:]

def get_stft(fn, num_frame):
    sig, fs = librosa.load(fn,sr=SAMPLING_RATE)
    assert(fs==SAMPLING_RATE)
    stft = e_stft(sig,WINDOW_LENGTH,HOG,'hann')
    stft = np.log10(np.abs(np.transpose(stft))+1e-7)
    remain = stft.shape[0]%num_frame
    #stft = np.concatenate((stft,stft[:remain,:]),axis = 0)
    return stft[:-remain,:]

def get_magnitude(magnitude, fn, num_frame):
    sig, fs = librosa.load(fn,sr=SAMPLING_RATE)
    n_sample = sig.shape[0]
    stft = e_stft(sig,WINDOW_LENGTH,HOG,'hann')
    abs_tf = np.abs(np.transpose(stft))
    remain = abs_tf.shape[0]%num_frame
    abs_tf = abs_tf[:-remain,:]
    #abs_tf = np.concatenate((abs_tf,abs_tf[:remain,:]),axis = 0)
    if magnitude is not None:
        magnitude = np.concatenate((magnitude,abs_tf),axis = 0)
    else:
        magnitude = abs_tf
    return magnitude

def get_feature(feat, fn, num_frame):
    tf= normalize_feat(fn, num_frame)
    if feat is not None:
        feat = np.concatenate((feat,tf),axis = 0)
    else:
        feat = tf
    return feat

def get_one_hot(target, fn, num_frame):
    tf_mix = get_stft(fn, num_frame)
    if tf_mix.shape[0]==0:
        return target
    fn_s1 = fn.replace('mix','s1')
    fn_s2 = fn.replace('mix','s2')
    tf1 = get_stft(fn_s1, num_frame)
    tf2 = get_stft(fn_s2, num_frame)
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


def generate_samples_chimera_net(f_list, batch_size=32, num_frame=100):
    #generate X * 100 * 129 feature
    # and     X * 100 * 3   label
    noisy_mag = None
    clean_s1 = None
    clean_s2 = None
    feat = None
    target = None
    while (feat is None or feat.shape[0]<num_frame*batch_size) and len(f_list)>0:
        #feature part
        fn = f_list.pop(0)
        f_list.append(fn)
        noisy_mag = get_magnitude(noisy_mag, fn, num_frame)
        clean_s1 = get_magnitude(clean_s1, fn.replace('mix','s1'), num_frame)
        clean_s2 = get_magnitude(clean_s2, fn.replace('mix','s2'), num_frame)
        feat = get_feature(feat, fn, num_frame)
        target = get_one_hot(target, fn, num_frame)
    if feat.shape[0]<batch_size*num_frame:
        return (None,None,None)
    feat = feat[0:batch_size*num_frame,:].reshape((-1,num_frame,129))
    target = target[0:batch_size*num_frame,:].reshape((-1,num_frame,129,NUM_SPEAKER))
    noisy_mag = noisy_mag[0:batch_size*num_frame,:].reshape((-1,num_frame,129))
    clean_s1 = clean_s1[0:batch_size*num_frame,:].reshape((-1,num_frame,129))
    clean_s2 = clean_s2[0:batch_size*num_frame,:].reshape((-1,num_frame,129))
    return (noisy_mag, clean_s1, clean_s2, feat, target)
