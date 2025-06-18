# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, AvgPool1d

class Time2Freq_Disc(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None,):
        super(Time2Freq_Disc, self).__init__()

        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.window_fn = torch.hann_window(win_len)

        if config is not None:
            self.use_compressed_input = config["use_compressed_input"]
            self.power = config["Train_cfg"]["power"]
        else:
            self.use_compressed_input = False
        self._eps = torch.tensor(1e-7)

    def forward(self, input_1d, oneframe = False):
        self._eps = self._eps.to(input_1d)
        self.window_fn = self.window_fn.to(input_1d)
        # input shape: (B,T)
        input_1d = input_1d.to(torch.float32) # to avoid float64-input
        if oneframe:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', center= False,return_complex=False).permute(0,3,2,1)
        else:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', return_complex=False).permute(0,3,2,1)   # (B,2,T,F)

        if self.use_compressed_input:
            mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2 + self._eps) ** 0.5
            mag_spec = mag_spec.unsqueeze(1)
            phs_ = stft_r / (mag_spec+self._eps)
            mag_spec_compressed = mag_spec ** self.power
            in_feature = mag_spec_compressed * phs_
        else:
            in_feature = stft_r

        return in_feature, stft_r
    
class Discriminator_Freq(nn.Module):
    def __init__(self, config=None, disc_cfg=None, sampling_rate=16000):
        super(Discriminator_Freq, self).__init__()
        frm_size = int(320 * sampling_rate / 16000) # 20ms
        shift_size = int(frm_size * 0.5)
        self.win_len = frm_size
        self.hop_len = shift_size
        n_fft = frm_size

        self.in_channels = 2
        activation = nn.LeakyReLU
        self.n_mfcc=13 if sampling_rate<48000 else 40
        self.sampling_rate = sampling_rate
        self.disc_cfg = disc_cfg
        
        ### input layer
        self.time2freq = Time2Freq_Disc(frm_size, shift_size, self.win_len, self.hop_len, n_fft, config)
        enc_kernel_size = (3, 2)

        ### encoder
        self.enc_time_kernel_len = enc_kernel_size[0]
        enc_channels = (8, 8, 16, 16)
        channel1d = 16*11
        enc_strides = (2, 2, 2, 2)
        pad = 1
        
        # conv layer
        chs_in = 2
        self.conv0 = nn.Conv2d(chs_in, enc_channels[0], enc_kernel_size, enc_strides[0], padding=pad)
        self.conv1 = nn.Conv2d(enc_channels[0], enc_channels[1], enc_kernel_size, enc_strides[1],padding=pad)
        self.conv2 = nn.Conv2d(enc_channels[1], enc_channels[2], enc_kernel_size, enc_strides[2],padding=pad)
        self.conv3 = nn.Conv2d(enc_channels[2], enc_channels[3], enc_kernel_size, enc_strides[3],padding=pad)
        self.out_conv = nn.Conv2d(channel1d, 1, kernel_size=1, stride=1, bias=False)

        self.in1 = nn.InstanceNorm2d(enc_channels[0],affine=True)
        self.in2 = nn.InstanceNorm2d(enc_channels[1],affine=True)
        self.in3 = nn.InstanceNorm2d(enc_channels[2],affine=True)
        self.in4 = nn.InstanceNorm2d(enc_channels[3],affine=True)
        # initial
        nn.init.xavier_normal_(self.conv0.weight)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.out_conv.weight)
        ## temporal pooling
        #linearInputChannels = 601 #601  # T*128
        self.avg_pooling = nn.AvgPool1d(10)  # 604/4
        if self.disc_cfg["adversarial_loss"] == 'BCE':
            self.activation = nn.Sigmoid()

    def forward(self, signal):
        # input shape: (B, L)
        in_feat, input_stft_r = self.time2freq(signal)
        ### encoder
        # 4 2D-conv layers
        self.feat_list = []  # for feat_loss on disc
        self.enc_out_list = [in_feat]
        feat0 = nn.LeakyReLU(0.1)(self.in1(self.conv0(in_feat)))
        feat1 = nn.LeakyReLU(0.1)(self.in2(self.conv1(feat0)))
        feat2 = nn.LeakyReLU(0.1)(self.in3(self.conv2(feat1)))
        feat3 = nn.LeakyReLU(0.1)(self.in4(self.conv3(feat2)))
        batch_size, nb_frames = feat3.shape[0], feat3.shape[2]
        feat3 = feat3.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
        out_feat = self.out_conv(feat3)
        logit = self.avg_pooling(out_feat.squeeze(-1))
        if self.disc_cfg["adversarial_loss"] == 'BCE':
            logit = self.activation(logit)
        logit = logit.squeeze(1) 
        self.feat_list=[feat0, feat1, feat2, feat3]
        return logit

class Discriminator(torch.nn.Module):   
    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config

        self.disc_cfg = config["Train_cfg"]["GAN_cfg"]
        self.disc_name = 'disc_freq'
        self.opt_name = 'opt_d_freq'
        self.dist_w = self.disc_cfg["d_freq_weight"]
        self.dist_w_g = self.disc_cfg["g_freq_weight"]
        self.dist_loss_g = "G_adv_freq"
        self.dist_loss_d = "D_freq"        

        self.discriminator = Discriminator_Freq(config=config, disc_cfg=self.disc_cfg, sampling_rate=config['sampling_rate'])

    def forward(self, signal, target):
        dis_fake = self.discriminator(signal)  
        dis_fake_feat_list = self.discriminator.feat_list  
        dis_real = self.discriminator(target)
        dis_real_feat_list = self.discriminator.feat_list           
                            
        return dis_fake, dis_real, dis_fake_feat_list, dis_real_feat_list        

    def my_state_dict(self, optimizer=None):
        dict = {self.disc_name: self.discriminator.state_dict()}
        if optimizer is not None:
            dict.update({self.opt_name: optimizer[1].optimizer.state_dict()})

        return dict
    
    def restore(self, model_caller, device, tmp_dict, optimizer=None):
        new_model_dict = self.discriminator.state_dict()
        if self.disc_name in tmp_dict.keys():
            model_dict = tmp_dict[self.disc_name]
        else:
            model_dict = {}
        new_dict = model_caller._restore_model(model_dict, 'discriminator')
        new_dict_opt = {k: v for k, v in new_dict.items() if k in new_model_dict}
        new_model_dict.update(new_dict_opt)
        self.discriminator.load_state_dict(new_model_dict)              

        if optimizer is not None:
            optimizer[1].optimizer.load_state_dict(tmp_dict[self.opt_name])
            for state in optimizer[1].optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            

