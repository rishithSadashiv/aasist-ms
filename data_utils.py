import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import sys
import librosa
sys.path.insert(0, './amplitude-modulation-analysis-module/')
from am_analysis import am_analysis as ama

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


def genSpoof_list_vops(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    vopStr_dict = {}
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label, vopStr = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            vopStr_dict[key] = vopStr
        return d_meta, file_list, vopStr_dict

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, vopStr = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
            vopStr_dict[key] = vopStr
        return file_list, vopStr_dict
    else:
        for line in l_meta:
            _, key, _, _, label, vopStr = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            vopStr_dict[key] = vopStr
        return d_meta, file_list, vopStr_dict


def repeat_padding(originalVops, n=10):
    paddedVops = []
    noOfVops = len(originalVops)
    
    if(noOfVops < n):
        i = 0
        paddedVops = originalVops
        while (noOfVops < n):
            paddedVops.append(originalVops[i])
            noOfVops += 1
            i += 1
            if(i == noOfVops):
                i = 0
    
    return paddedVops

def zero_padding(segment, n):
    # print(len(segment))
    length = len(segment)
    result = np.zeros(n)
    # print(result.shape)
    result[:length] = segment
    return result

def getIndices(vop):
    indices = [ind for ind, ele in enumerate(vop) if ele == 1]
    return indices

def modulation_spectogram_from_wav(audio_data,fs):
    x=audio_data
    x = x / np.max(x)
    win_size_sec = 0.015 
    win_shft_sec = 0.005  
    # print('before am analysis')
    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(x, fs, win_size = round(win_size_sec*fs), win_shift = round(win_shft_sec*fs), channel_names = ['Modulation Spectrogram'])
    
    X_plot=ama.plot_modulation_spectrogram_data(stft_modulation_spectrogram, 0 , modf_range = np.array([0,20]), c_range =  np.array([-90, -50]))
    # print('after xplot', X_plot.shape)
    return X_plot

def load_data(segment, sr=8000):
    linear_spect = modulation_spectogram_from_wav(segment,sr)
    # print('linear_spect')
    mag_T = linear_spect

    mu = np.mean(mag_T)
    std = np.std(mag_T)
    return (mag_T - mu) / (std + 1e-5)

def getFeatures(path, vops):
    fs = 8000
    ms50 = int(0.05 * fs)
    features = np.zeros((1,61,10))
    audio, _ = librosa.load(path, sr=fs)
    fileVopIndices = vops.split(',')
    fileVopIndices = [ int(x) for x in fileVopIndices ]
    n = 10
    if(len(fileVopIndices) < n):
        fileVopIndices = repeat_padding(fileVopIndices)
    else:
        fileVopIndices = fileVopIndices[:n]
        
    for vopIndex in fileVopIndices:
        # print(vopIndex)
        startPoint = max(0, vopIndex - ms50)
        # print(np.shape(audio))
        # print(vopIndex + ms50)
        endPoint = min(np.shape(audio)[0], vopIndex + ms50)
        # print(startPoint)
        # print(endPoint)
        segment100ms = audio[startPoint: endPoint]
        # print(len(segment100ms))
        if((endPoint - startPoint) < 2*ms50):
            # print(endPoint - startPoint)
            segment100ms = zero_padding(segment100ms, 2*ms50)
        ms = load_data(segment100ms)
        # print(ms.shape)
        # print('feature extracted: ', ms.shape)
        features = np.concatenate((features, ms.reshape(1, 61, 10)), axis = 0)
        # print(features.shape)
    # print('features extracted:', features.shape)
    return features[1:, :]


class customTrainMS(Dataset):
    def __init__(self, list_IDs, labels, base_dir, vopStr_dict):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.vopStr_dict = vopStr_dict
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        vops = self.vopStr_dict[key]
        # print(vops)
        # print(vops[0])
        # fileVopIndices = vops.split(',')
        # fileVopIndices = [ int(a) for a in fileVopIndices ]
        path = str(self.base_dir / f"flac/{key}.flac")
        x = getFeatures(path, vops)
        x = x.swapaxes(0, 1).reshape(x.shape[1], -1)
        print(x.shape)
        # X_pad = pad_random(X, self.cut)
        # x_inp = Tensor(X_pad)
        x_inp = Tensor(x)
        y = self.labels[key]
        return x_inp, y


class customDevNevalMS(Dataset):
    def __init__(self, list_IDs, base_dir, vopStr_dict):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.vopStr_dict = vopStr_dict
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        vops = self.vopStr_dict[key]
        
        # fileVopIndices = vops.split(',')
        # fileVopIndices = [ int(a) for a in fileVopIndices ]
        path = str(self.base_dir / f"flac/{key}.flac")
        x = getFeatures(path, vops)
        x = x.swapaxes(0, 1).reshape(x.shape[1], -1)
        print(x.shape)
        # X_pad = pad(X, self.cut)
        # x_inp = Tensor(X_pad)
        x_inp = Tensor(x)
        return x_inp, key
