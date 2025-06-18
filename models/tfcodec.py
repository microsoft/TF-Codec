# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class TFCodec(MultiFrmVQBottleNeck):
    def __init__(self, config=None):
        super().__init__(config=config, bitrate=config['bitrate'], sampling_rate=config["sampling_rate"], feat_dim=320)   

        self.sampling_rate = config["sampling_rate"]
        frm_size = config["dft_size"]
        shift_size = int(config["dft_size"] * config["hop_vqvae"])
        self.win_len = frm_size
        self.hop_len = shift_size
        self.frm_size = frm_size        

        self.in_channels = 2
        activation_functions = {'PRELU':nn.PReLU,'ELU':nn.ELU}
        activation = activation_functions['PRELU']
        self.config = config
        
        if config["use_learnable_compression"]:
            self.power_cprs = nn.Parameter(torch.FloatTensor(1))
            nn.init.constant_(self.power_cprs, 0.4)#config["power"])

        ### input layer
        self.time2freq = Time2Freq2(frm_size, shift_size, self.win_len, self.hop_len, self.win_len, config=config, power=self.power_cprs if config["use_learnable_compression"] else config["input_cprs_power"])

        ### encoder
        if frm_size == 320:  # 20ms
            enc_kernel_size = (2, 5)            
            enc_channels = (16, 32, 64, 64)
            enc_strides = ((1, 1), (1, 4), (1, 4), (1, 2)) # 161, 41, 11, 5
            self.frm_pad = ((2, 2), (2, 2), (2, 2), (1, 1))
            self.last_layer_pad = (2, 2)        

            self.enc_out_frm_size = 5
            self.enc_out_channels = enc_channels[-1]
            self.tcm_mid_channels = 512
            self.tcm_repeat1, self.tcm_block1 = 1, 4
            self.gru_num = 1
            self.tcm_repeat2, self.tcm_block2 = 1, 4
            self.depth_ker_size = 5
            
            self.enc_time_kernel_len = enc_kernel_size[0]
            is_last = [False, False, False, True]
            self.pad_list = [None, None, None, (self.frm_pad[0], (self.enc_time_kernel_len - 1, 0))]
        elif frm_size == 640:  # 40ms
            enc_kernel_size = (2, 5)
            enc_channels = (16, 32, 32, 64, 64)
            enc_strides = ((1, 1), (1, 2), (1, 4), (1, 4), (1, 2)) #321, 161, 41, 11, 5
            self.frm_pad = ((2, 2), (2, 2), (2, 2), (2, 2), (1, 1))
            self.last_layer_pad = (2, 2)        

            self.enc_out_frm_size = 5
            self.enc_out_channels = enc_channels[-1]
            self.tcm_mid_channels = 512
            self.tcm_repeat1, self.tcm_block1 = 1, 4
            self.gru_num = 1
            self.tcm_repeat2, self.tcm_block2 = 1, 4
            self.depth_ker_size = 3
            
            self.enc_time_kernel_len = enc_kernel_size[0]
            is_last = [False, False, False, False, True]
            self.pad_list = [None, None, None, None, (self.frm_pad[0], (self.enc_time_kernel_len - 1, 0))]

        self.feat_dim = self.enc_out_frm_size * self.enc_out_channels
        assert(self.feat_dim == 320)  # used for VQ module initialization
        
        self.enc_list = nn.ModuleList(
            ConvLayer(in_c, out_c, enc_kernel_size, enc_strides[i], nonlinearity=activation)
            for i, (in_c, out_c) in enumerate(
                zip((self.in_channels,) + enc_channels[:-1], enc_channels)))
                
        # encoder tcm
        self.tcm_list_enc = nn.ModuleList(TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation)
                                        for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))

        self.gru_enc = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)
        self.h0_enc = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())
        
        ### tcm-gru-tcm
        self.tcm_list1 = nn.ModuleList(TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation)
                                       for _ in range(self.tcm_repeat1) for i in range(self.tcm_block1))
        self.gru = nn.GRU(self.feat_dim, self.feat_dim, num_layers=self.gru_num, bias=True, batch_first=True)        
        self.h0 = nn.Parameter(torch.FloatTensor(self.gru_num, 1, self.feat_dim).zero_())

        self.tcm_list2 = nn.ModuleList(TCMLayer(self.feat_dim, self.tcm_mid_channels, self.depth_ker_size, 2 ** i, nonlinearity=activation)
                                           for _ in range(self.tcm_repeat2) for i in range(self.tcm_block2))

        ### decoder        
        out_dim = 2
        self.dec_list = nn.ModuleList(
            DeconvLayer(in_c, out_c, enc_kernel_size,enc_strides[len(self.enc_list) - 1 - i], nonlinearity=activation, is_last=is_last[i])
            for i, (in_c, out_c) in enumerate(
                zip(enc_channels[::-1], enc_channels[::-1][1:] + (out_dim,)))
        )

        ### last layer
        self.out_conv = nn.Conv2d(out_dim, out_dim, enc_kernel_size, (1, 1), bias=False)
        nn.init.xavier_normal_(self.out_conv.weight)
        self.freq2time = Freq2TimeCodec(frm_size, shift_size, self.win_len, self.hop_len, self.win_len, power=self.power_cprs if config["use_learnable_compression"] else config["input_cprs_power"], config=config)    

    def load(self, ckpt_path): 
        new_dict = {}
        current_model_dict = self.state_dict()

        if ckpt_path is not None:
            print('Model loaded from {}'.format(ckpt_path))
            tmp_dict = torch.load(ckpt_path, map_location='cpu')
            tmp_dict2 = tmp_dict["gen"] if 'gen' in tmp_dict.keys() else tmp_dict        
            print('keys to load:{}'.format(len(tmp_dict2.keys())))
            for key in tmp_dict2.keys():
                new_key = key.split('module.')[-1]
                if 'generator.' in new_key:
                    new_key = new_key.split('generator.')[-1]
                new_dict[new_key] = tmp_dict2[key] 
        self.load_state_dict(new_dict, strict=True)
        print('keys loaded:{}'.format(len(new_dict.keys())))

    def forward(self, inputs):
        self.bitrate = self.config["bitrate"]
        vq_result = self.encode_to_token_idx(inputs)

        vq_feat = vq_result["vq_feat"]             
        pred_1d, pred_2d = self.decoder(vq_feat) 
        if self.pad_len > 0:
            pred_1d = pred_1d[:,:-self.pad_len]                  

        result = {} 
        result[self.bitrate] = {}
        result[self.bitrate].update(vq_result)
        result[self.bitrate]["x_hat"] = pred_1d
        return result      

    def encode_to_token_idx(self, inputs):        
        self.signal = inputs
        self.signal, self.pad_len = rpad_signal_for_codec(self.signal, self.combine_frames, self.hop_len)
                
        in_feat, self.input_stft_r = self.time2freq(self.signal)  
        
        enc_feat = self.encoder(in_feat)  
        
        vq_result = self.vq_bottleneck(enc_feat, loss_mask=None, epo=None)

        return vq_result 

    def decode_from_token_idx(self, vq_inds): 
        vq_feat = self.decode_vq_bottleneck(vq_inds)        
        pred_1d, _ = self.decoder(vq_feat) 
        return pred_1d         

    def encoder(self, in_feat):
        x_conv = in_feat  # B,2,T,161

        ### encoder
        self.enc_out_list = [x_conv]
        for layer_i in range(len(self.enc_list)):
            enc_layer = self.enc_list[layer_i]
            enc_out = enc_layer(self.enc_out_list[-1], (self.frm_pad[layer_i],(self.enc_time_kernel_len-1,0)))
            self.enc_out_list.append(enc_out)    

        enc_out = self.enc_out_list[-1]  #B,64,T,5
        batch_size, nb_frames = enc_out.shape[0], enc_out.shape[2]
        enc_out = enc_out.permute(0, 2, 1, 3).reshape(batch_size, nb_frames, -1).permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
                  
        tcm_feat = enc_out
        for tcm in self.tcm_list_enc:
            tcm_feat ,_= tcm(tcm_feat)
        enc_out = tcm_feat

        rnn_in = enc_out
        rnn_in = rnn_in.squeeze(-1).permute(0, 2, 1)  # (B,T,C)
        rnn_out, h_n_enc = self.gru_enc(rnn_in, self._gru_init_state_enc(batch_size))
        rnn_out = rnn_out.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)
        enc_out = rnn_out

        return enc_out

    def decoder(self, vq_feat):        
        ### interleave structure
        batch_size, nb_frames = vq_feat.shape[0], vq_feat.shape[2]  # (B,C,T,1)
        tcm_feat = vq_feat
        for tcm in self.tcm_list1:
            tcm_feat,_ = tcm(tcm_feat)        
 
        tcm_feat = tcm_feat.squeeze(-1).permute(0, 2, 1)  # (B,T,C)
        tcm_feat, h_n = self.gru(tcm_feat, self._gru_init_state(batch_size))
        tcm_feat = tcm_feat.permute(0, 2, 1).unsqueeze(-1)  # (B,C,T,1)

        for tcm in self.tcm_list2:
            tcm_feat,_ = tcm(tcm_feat)

        dec_input = tcm_feat.squeeze(-1).reshape(batch_size, self.enc_out_channels, self.enc_out_frm_size, nb_frames).permute(0, 1, 3, 2)        
        self.dec_out_list = [dec_input]
        for layer_i, dec_layer in enumerate(self.dec_list):
            dec_input = dec_layer(dec_input, None, self.frm_pad[::-1][layer_i], pad_size=self.pad_list[layer_i])
            self.dec_out_list.append(dec_input)

        dec_out = dec_input  # (B,2,T,161)

        dec_out = F.pad(dec_out, (self.last_layer_pad[0], self.last_layer_pad[1], self.enc_time_kernel_len - 1, 0))
        x_out = self.out_conv(dec_out)
        pred_1d = self.freq2time(x_out)               
            
        return pred_1d, x_out

    def _gru_init_state(self, n):
        if not torch._C._get_tracing_state():
            return self.h0.expand(-1, n, -1).contiguous()
        else:
            return self.h0.expand(self.h0.size(0), n, self.h0.size(2)).contiguous()

    def _gru_init_state_enc(self, n):
        if not torch._C._get_tracing_state():
            return self.h0_enc.expand(-1, n, -1).contiguous()
        else:
            return self.h0_enc.expand(self.h0_enc.size(0), n, self.h0_enc.size(2)).contiguous()

    
    