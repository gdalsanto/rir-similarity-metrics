import torch 
import torch.nn.functional as F
import torch.nn as nn
import scipy.signal as signal
import numpy as np 
import auraloss
from utils import window2d
from filterbank import FilterBank
import torch


import torch.nn as nn
import torch.nn.functional as F

class MAE_stft(nn.Module):
    def __init__(self, n_fft=1024, hop_size=0.25, fs=44100):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.fs = fs

    def forward(self, y_pred, y_true):
        '''compute the mean absolute error between the STFT of two RIRs'''

        # compute the STFT using torch
        Zxx1 = torch.stft(y_pred.squeeze(), n_fft=self.n_fft, hop_length=int(self.n_fft*self.hop_size), return_complex=True)
        Zxx2 = torch.stft(y_true.squeeze(), n_fft=self.n_fft, hop_length=int(self.n_fft*self.hop_size), return_complex=True)

        # compute the mean absolute error
        return torch.mean(torch.abs(torch.abs(Zxx1) - torch.abs(Zxx2)))

class MultiResoSTFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.MRstft = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, y_pred, y_true):
        '''compute the Multi Scale Spectral loss using auraloss'''
        return self.MRstft(y_pred, y_true)


class ESRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ESR = auraloss.time.ESRLoss(reduction='mean')

    def forward(self, y_pred, y_true):
        '''compute the error to signal ratio using auraloss'''
        return self.ESR(y_pred, y_true)


class AveragePower(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        '''distance of the average power of two signals'''
        # compute the magnitude spectrogram 
        # ungly unsequeezing is necessary to make the shapes compatible TODO fix this when implemented in a DL framework 
        S1 = torch.pow(torch.abs(torch.stft(y_pred.squeeze(), n_fft=1024, hop_length=256, return_complex=True)),2).unsqueeze(0).unsqueeze(0)
        S2 = torch.pow(torch.abs(torch.stft(y_true.squeeze(), n_fft=1024, hop_length=256, return_complex=True)),2).unsqueeze(0).unsqueeze(0)
        
        # create 2d window
        win = window2d(torch.hann_window(64, dtype=S1.dtype)).unsqueeze(0).unsqueeze(0)
        # convove spectrograms with the window
        S1_win = F.conv2d(S1, win, stride=(4, 4)).squeeze()
        S2_win = F.conv2d(S2, win, stride=(4, 4)).squeeze()
        # compute the normalized difference between the two windowed spectrograms 
        return torch.norm(S2_win - S1_win, p="fro") / torch.norm(S2_win, p="fro") / torch.norm(S1_win, p="fro")

class EDCLoss(nn.Module):
    """
    EDCLoss: custom loss function for computing the normalized mean squared error 
    on the Energy Decay Curves (EDCs).
    """
    def __init__(self, backend='torch', sr=48000, nfft=None):
        super().__init__()   

        self.sr = sr 
        self.filterbank = FilterBank(fraction=3, 
                                order=5, 
                                fmin=60, 
                                fmax=6000, 
                                sample_rate=self.sr, 
                                backend=backend,
                                nfft=nfft)
        self.mse = nn.MSELoss(reduction='mean')

    def discard_last_n_percent(self, edc, n_percent):
        '''Discards the last n_percent of the EDC.'''
        last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
        out = edc[..., 0:last_id]

        return out
    
    def backward_int(self, x):
        '''Performs a Schroeder backwards integral on the last dimension of the 
        input tensor.'''
        x = torch.flip(x, [-1])
        x = (1 / x.shape[-1]) * torch.cumsum(x ** 2, -1)
        return torch.flip(x, [-1])


    def forward(self, y_pred, y_true):
        '''Computes the normalized mean squared error on the EDCs.'''
        # Remove filtering artefacts (last 5 permille)
        y_pred = self.discard_last_n_percent(y_pred.squeeze(0), 0.5)
        y_true = self.discard_last_n_percent(y_true.squeeze(0), 0.5)
        # compute EDCs
        y_pred_edr = self.backward_int(self.filterbank(y_pred))
        y_true_edr = self.backward_int(self.filterbank(y_true))
        y_pred_edr = 10*torch.log10(y_pred_edr + 1e-32)
        y_true_edr = 10*torch.log10(y_true_edr + 1e-32)
        level_pred = y_pred_edr[:,:,0]
        level_true = y_true_edr[:,:,0]
        # compute normalized mean squared error on the EDCs 
        num = self.mse(y_pred_edr - level_pred.unsqueeze(-1), y_true_edr - level_true.unsqueeze(-1))
        den = torch.mean(torch.pow(y_true_edr - level_true.unsqueeze(-1), 2))
        return  num / den
        