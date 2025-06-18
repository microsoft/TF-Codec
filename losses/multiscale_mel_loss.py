# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torchaudio.transforms import MelScale
import math

class MultiscaleMelLoss(object):
    def __init__(self, config, sampling_rate, rank=0) -> None:
        sampling_scale = sampling_rate // 16000
        f_min = int(20*sampling_scale)
        f_max = int(8000*sampling_scale)
        self.config = config
        if torch.cuda.is_available():
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        self.eps = torch.tensor(1e-7).to(device)
        self.mse_loss = nn.MSELoss().cuda(device)
        self.mae_loss = nn.L1Loss().cuda(device)   

        ## multiscale on window_len
        if config["Train_cfg"]["mel_loss_type"] == 'multi_window':
            self.multi_win_len = [128,256,512,1024,2048]*sampling_scale #[64,128,256,512,1024,2048]*sampling_scale
            self.multi_n_fft = self.multi_win_len
            self.multi_window_fn = [torch.hann_window(win_len).to(device) for win_len in self.multi_win_len]
            if sampling_rate == 24000:
                self.mel_scale_list1 = [MelScale(n_mels=64,sample_rate=sampling_rate,f_min=f_min,f_max=f_max, n_stft=n_fft//2+1).to(device) for n_fft in self.multi_n_fft]
            else:
                self.mel_scale_list1 = [MelScale(n_mels=64*sampling_scale,sample_rate=sampling_rate,f_min=f_min,f_max=f_max, n_stft=n_fft//2+1).to(device) for n_fft in self.multi_n_fft]

        self.hop_vqvae = 0.25

        ## multiscale on n_mel_banks
        if config["Train_cfg"]["mel_loss_type"] == 'multi_band':
            multi_n_mels = [64,128]*sampling_scale
            self.n_fft = 512*sampling_scale
            self.win_len = self.n_fft
            self.window_fn = torch.hann_window(self.win_len).to(device)
            self.hop_len = int(self.win_len*self.hop_vqvae)
            self.mel_scale_list2 = [MelScale(n_mels=n_mels,sample_rate=sampling_rate,f_min=f_min,f_max=f_max,n_stft=self.n_fft//2+1).to(device) for n_mels in multi_n_mels]


    def compute_loss_multi_band(self, pred_1d, target_1d, ):
        n_fft = self.n_fft
        hop_len = self.hop_len
        win_len = n_fft
        window_fn = self.window_fn
        eps = self.eps

        # input shape: (B,L)
        pred_stft_r = torch.stft(pred_1d, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window_fn,pad_mode='constant', return_complex=False)
        pred_mag = (pred_stft_r[:, :, :, 0] ** 2 + pred_stft_r[:, :, :, 1] ** 2 + eps) ** 0.5
        target_stft_r = torch.stft(target_1d, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window_fn,pad_mode='constant', return_complex=False)  # (B,F,T,2)
        target_mag = (target_stft_r[:, :, :, 0] ** 2 + target_stft_r[:, :, :, 1] ** 2 + eps) ** 0.5
        mel_loss = 0
        for mel_scale in self.mel_scale_list2:
            pred_mel = mel_scale(pred_mag)
            target_mel = mel_scale(target_mag)
            mel_loss += self.mse_loss(pred_mel,target_mel)           
        mel_loss /= len(self.mel_scale_list2)        

        return mel_loss

    def compute_loss_multi_win(self, pred_1d, target_1d, ):
        eps = self.eps
        mel_loss = 0
        # input shape: (B,L)
        for win_len,n_fft,window_fn,mel_scale in zip(self.multi_win_len,self.multi_n_fft,self.multi_window_fn,self.mel_scale_list1):
            hop_len = int(win_len * self.hop_vqvae)
            pred_stft_r = torch.stft(pred_1d, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window_fn,pad_mode='constant', return_complex=False)
            pred_mag = (pred_stft_r[:, :, :, 0] ** 2 + pred_stft_r[:, :, :, 1] ** 2 + eps) ** 0.5
            target_stft_r = torch.stft(target_1d, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window_fn,pad_mode='constant', return_complex=False)  # (B,F,T,2)
            target_mag = (target_stft_r[:, :, :, 0] ** 2 + target_stft_r[:, :, :, 1] ** 2 + eps) ** 0.5
            pred_mel = mel_scale(pred_mag)
            target_mel = mel_scale(target_mag)
            mel_loss += self.mae_loss(pred_mel, target_mel)
            if self.config["Train_cfg"]["use_log_L2_term"]:
                mel_loss += self.config["Train_cfg"]["mel_L2_term_weight"]*math.sqrt(win_len/2.0)*self.mse_loss(torch.log(pred_mel + eps), torch.log(target_mel + eps))
        mel_loss /= len(self.multi_win_len)

        return mel_loss

    def compute_loss(self,pred_1d, target_1d):
        if self.config["Train_cfg"]["mel_loss_type"] == 'multi_window':
            loss = self.compute_loss_multi_win(pred_1d,target_1d)
        if self.config["Train_cfg"]["mel_loss_type"] == 'multi_band':
            loss = self.compute_loss_multi_band(pred_1d,target_1d)
        return loss
