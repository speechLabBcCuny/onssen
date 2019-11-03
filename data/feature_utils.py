import librosa
import numpy as np


def get_stft(fn, sampling_rate, window_size, hop_size):
    """
    fn: the absolute path of the wav file
    sampling_rate: in Hz
    window_size: window size for fft
    hop_size: the hop size for shifting the window

    return:
        stft: frame * frequency numpy array
    """
    sig, fs = librosa.load(fn, sr = None)
    if fs != sampling_rate:
        # print("WARNING!!! The sampling rate provided is different from the data")
        # print("Resample the audio...")
        sig = librosa.core.resample(sig, fs, sampling_rate)
    stft = np.transpose(librosa.core.stft(sig, n_fft=window_size, hop_length=hop_size))
    return stft


def get_stft_from_subtraction(f_mix, f_clean, sampling_rate, window_size, hop_size):
    sig_mix, fs = librosa.load(f_mix, sr = None)
    sig_clean, fs = librosa.load(f_clean, sr = None)
    sig_noise = sig_mix - sig_clean
    if fs != sampling_rate:
        # print("WARNING!!! The sampling rate provided is different from the data")
        # print("Resample the audio...")
        sig_noise = librosa.core.resample(sig_noise, fs, sampling_rate)
    stft = np.transpose(librosa.core.stft(sig_noise, n_fft=window_size, hop_length=hop_size))
    return stft


def get_log_mel_spectrogram(fn, sampling_rate, window_size, hop_size, epsilon=1e-7):
    sig, fs = librosa.load(fn, sr = None)
    assert sampling_rate == fs
    mel_spectra = librosa.feature.melspectrogram(
        sig,
        sr=sampling_rate,
        n_fft=window_size,
        hop_length=hop_size
    )
    mel_spectra = np.transpose(np.log10(mel_spectra + epsilon))
    mel_spectra = librosa.util.normalize(mel_spectra, axis=1)
    return mel_spectra


def get_log_magnitude(stft, epsilon=1e-7):
    feature = np.log10(np.abs(stft) + epsilon)
    feature = librosa.util.normalize(feature, axis=1)
    return feature


def get_phase(stft):
    """
    stft: frame * frequency complex numpy array
    return:
        phase: frame * frequency * 2 real numpy array
    """
    real = np.real(stft)
    imag = np.imag(stft)
    phase = np.array([real, imag])
    phase = np.transpose(phase, (1,2,0))
    return phase


def get_angle(stft):
    """
    stft: frame * frequency complex numpy array
    return:
        angle: the angle of the STFT
    """
    angle = np.angle(stft)
    return angle


def get_cos_difference(stft_1, stft_2):
    angle_1 = get_angle(stft_1)
    angle_2 = get_angle(stft_2)
    return np.cos(angle_1 - angle_2)


def get_one_hot(feature_mix, mag_s1, mag_s2, db_threshold):
    specs = np.asarray([mag_s1, mag_s2])
    vals = np.argmax(specs, axis=0)
    Y = np.zeros(mag_s1.shape+(2,))
    for i in range(2):
        temp = np.zeros((2))
        temp[i]=1
        Y[vals == i] = temp
    #label the silence part
    m = np.max(feature_mix) - db_threshold/20
    temp = np.zeros((2))
    Y[feature_mix < m] = temp
    return Y
