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
import scipy
import numpy as np
from scipy.io import loadmat

def normalize_feat(fn):
    abs_tf = np.log10(np.abs(np.transpose(loadmat(fn)['stft']))+1e-7)
    return abs_tf

def get_magnitude(magnitude, fn):
    tf = np.abs(np.transpose(scipy.io.loadmat(fn)['stft']))
    if magnitude is not None:
        magnitude = np.concatenate((magnitude,tf),axis = 0)
    else:
        magnitude = tf
    return magnitude

def get_feature(feat, fn):
    tf = normalize_feat(fn)
    if feat is not None:
        feat = np.concatenate((feat,tf),axis = 0)
    else:
        feat = tf
    return feat

def get_one_hot(target, fn):
    tf_mix = normalize_feat(fn)
    fn_s1 = fn.replace('mix','s1')
    fn_s2 = fn.replace('mix','s2')
    tf1 = normalize_feat(fn_s1)
    tf2 = normalize_feat(fn_s2)
    #do we need normalize the spectrogram?
    specs = np.asarray([tf1, tf2])
    vals = np.argmax(specs, axis=0)
    Y = np.zeros(tf1.shape+(3,))
    for i in range(2):
        temp = np.zeros((3))
        temp[i]=1
        Y[vals == i] = temp
    #label the silence part
    m = np.max(tf_mix) - 40/20
    temp = np.zeros((3))
    # temp[2]=1
    Y[tf_mix < m] = temp
    if target is None:
        return Y
    else:
        return np.concatenate((target,Y),axis = 0)


def generate_samples(f_list, feat = None, target = None):
    #generate X * 100 * 129 feature
    # and     X * 100 * 3   label
    while (feat is None or feat.shape[0]<3200) and len(f_list)>0:
        #feature part
        fn = f_list.pop(0)
        feat = get_feature(feat, fn)
        target = get_one_hot(target, fn)
    return feat, target

def generate_samples_chimera_net(f_list, batch_size=32, magnitude = None, feat = None, target = None):
    #generate X * 100 * 129 feature
    # and     X * 100 * 3   label
    while (feat is None or feat.shape[0]<100*batch_size) and len(f_list)>0:
        #feature part
        fn = f_list.pop(0)
        magnitude = get_magnitude(magnitude, fn)
        feat = get_feature(feat, fn)
        target = get_one_hot(target, fn)
    if feat.shape[0]<batch_size*100:
        return (None,None,None),(None,None,None)
    inputs = feat[0:batch_size*100,:]
    labels = target[0:batch_size*100,:,:]
    mag = magnitude[0:batch_size*100,:]
    mag = mag.reshape((batch_size,100,129))
    inputs = inputs.reshape((batch_size,100,129))
    labels = labels.reshape((batch_size,100,129,3))
    magnitude = magnitude[batch_size*100:,:]
    feat = feat[batch_size*100:,:]
    target = target[batch_size*100:,:,:]
    return (mag, inputs, labels),(magnitude, feat, target)
