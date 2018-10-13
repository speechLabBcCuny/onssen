import glob
import os
import librosa
from sklearn.cluster import KMeans
from stft_utils import e_stft, e_istft
from feat_generator import get_one_hot, get_stft, normalize_feat
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from separation import bss_eval_sources

def generate_file_list(path):
    file_list = glob.glob(path)
    result = []
    for ele in file_list:
        result.append(os.path.basename(ele))
    return result

def get_test_stft(fn):
    sig, fs = librosa.load(fn,sr=8000)
    assert(fs==8000)
    stft = e_stft(sig,256,64,'hamming')
    #stft = np.concatenate((stft,stft[:remain,:]),axis = 0)
    return stft

def get_test_normalize_feat(fn):
    sig, fs = librosa.load(fn,sr=8000)
    assert(fs==8000)
    # sig = sig-np.mean(sig)
    # sig = sig/(np.max(np.abs(sig))) +1e-7
    n_sample = sig.shape[0]
    stft = e_stft(sig,256,64,'hamming')
    return np.log10(np.abs(np.transpose(stft))+1e-7)

def evaluate_separation(test_list):
    SDR_SUM = 0.0
    SIR_SUM = 0.0
    SAR_SUM = 0.0
    m = torch.load('./dc_models/32_model')
    m.cpu()
    m.cuda()
    for ele in test_list:
        fn_mix = '/scratch/near/2speakers/wav8k/min/tt/mix/'+ele
        fn_s1 = '/scratch/near/2speakers/wav8k/min/tt/s1/'+ele
        fn_s2 = '/scratch/near/2speakers/wav8k/min/tt/s2/'+ele
        feature = get_test_normalize_feat(fn_mix)
        feature = feature.reshape(1,-1,129)
        device = torch.device('cuda')
        feature = torch.tensor(feature,dtype=torch.float,device=device)
        embedding = m(feature)
        embedding
        embedding = embedding.reshape(1,-1,129,40)
        silence_mask = torch.max(feature)-2
        kmeans_input = embedding[feature>silence_mask]
        kmeans_input = kmeans_input.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=2, random_state=0).fit(kmeans_input)
        kmean_mask = kmeans.predict(kmeans_input)
        mask_s1 = torch.zeros((feature.shape),dtype=torch.int)
        mask_s2 = torch.zeros((feature.shape),dtype=torch.int)
        mask_s1[feature>silence_mask] = torch.tensor(kmean_mask,dtype=torch.int)
        mask_s2[feature>silence_mask] = torch.tensor(1-kmean_mask,dtype=torch.int)
        mask_s1 = mask_s1.reshape(-1,129)
        mask_s2 = mask_s2.reshape(-1,129)
        mask_s1 = mask_s1.detach().numpy()
        mask_s2 = mask_s2.detach().numpy()
        s1, fs = librosa.load(fn_s1,sr=8000)
        s2, fs = librosa.load(fn_s2,sr=8000)
        stft = get_test_stft(fn_mix)
        stft_s2 = stft*mask_s1.T
        stft_s1 = stft*mask_s2.T
        enhanced_s1 = e_istft(stft_s1, s1.shape[0],256, 64, 'hamming')
        enhanced_s2 = e_istft(stft_s2, s1.shape[0],256, 64, 'hamming')
        sdr, sir, sar, _ = bss_eval_sources(np.array([s1,s2]),np.array([enhanced_s1,enhanced_s2]))
        file_path = './evaluation_results/32_model/%s.txt'%ele
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(file_path,'w')
        f.write('SDR: %f %f\n'%(sdr[0],sdr[1]))
        f.write('SIR: %f %f\n'%(sir[0],sir[1]))
        f.write('SAR: %f %f\n'%(sar[0],sar[1]))
        f.close()
        SDR_SUM+= sdr[0]+sdr[1]
        SIR_SUM+=sir[0]+sir[1]
        SAR_SUM+=sar[1]+sar[1]
    print("SDR: %f"%(SDR_SUM/(len(test_list)*2)))
    print("SIR: %f"%(SIR_SUM/(len(test_list)*2)))
    print("SAR: %f"%(SAR_SUM/(len(test_list)*2)))




test_list = '/scratch/near/2speakers/wav8k/min/tt/s1/*.wav'
test_list = generate_file_list(test_list)
evaluate_separation(test_list)
print("Done")
