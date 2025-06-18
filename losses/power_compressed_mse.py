# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn

class PowerCompressedMSE(object):
    def __init__(self, config, rank=0) -> None:
        self.weight_amplitude = config['Train_cfg']['weight_amplitude']
        self.power = config['Train_cfg']['power']        
        if torch.cuda.is_available():
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")    
        self.eps = torch.tensor(1e-7).to(device)
        self.mse_loss = nn.MSELoss().cuda(device)
        self.device = device
        self.config = config

    def compute_loss(self, pred_1d, target_1d, sampling_rate):        
        eps = self.eps
        win_len = int(sampling_rate*self.config['frame_dur'])
        hop_len = int(win_len*0.5)        
        n_fft = win_len
        window_fn = torch.hann_window(win_len).to(self.device)

        # input shape: (B,L)
        pred_stft_r = torch.stft(pred_1d, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window_fn, 
                pad_mode='constant', return_complex=False)
        target_stft_r = torch.stft(target_1d, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window_fn, 
                pad_mode='constant', return_complex=False) # (B,F,T,2)
        target_mag = (target_stft_r[:,:,:,0]**2 + target_stft_r[:,:,:,1]**2 + eps) ** 0.5
        target_phase = target_stft_r / (target_mag.unsqueeze(-1) + eps)
        pred_mag = (pred_stft_r[:,:,:,0]**2 + pred_stft_r[:,:,:,1]**2 + eps) ** 0.5
        pred_phase = pred_stft_r / (pred_mag.unsqueeze(-1) + eps)

        # compute the amplitude loss
        compressed_target_mag = torch.pow(target_mag + eps, self.power)
        compressed_estim_mag = torch.pow(pred_mag + eps, self.power)
        loss_mag = self.mse_loss(compressed_estim_mag, compressed_target_mag)

        # compute the whole spectrogram (phase-aware) loss
        estim_real = compressed_estim_mag * pred_phase[:,:,:,0]
        estim_imag = compressed_estim_mag * pred_phase[:,:,:,1]
        target_real = compressed_target_mag * target_phase[:,:,:,0]
        target_imag = compressed_target_mag * target_phase[:,:,:,1]
        loss_spec = 0.5* self.mse_loss(estim_real, target_real) + 0.5* self.mse_loss(estim_imag, target_imag)

        return loss_mag * self.weight_amplitude + loss_spec * (1 - self.weight_amplitude)

    def _get_amp_and_phase(self, x_out):
        x_out_real = x_out[:, 0, :, :]  # (B,T,F)
        x_out_imag = x_out[:, 1, :, :]
        amp = (x_out_real ** 2 + x_out_imag ** 2 + self.eps) ** 0.5
        amp = amp.unsqueeze(1) # (B,2,T,F)
        phase = x_out / (amp + self.eps)
        return amp, phase