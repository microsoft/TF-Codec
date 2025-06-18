# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
import numpy as np

class Time2Freq2(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None, power=None):
        super(Time2Freq2, self).__init__()

        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.window_fn = torch.hann_window(win_len)
        self.config = config
        if config is not None:
            self.use_compressed_input = config["use_compressed_input"]
            self.power = config["input_cprs_power"] if power is None else power
        else:
            self.use_compressed_input = False
        self._eps = torch.tensor(1e-7).to(torch.float32)

    def forward(self, input_1d, oneframe = False):
        self._eps = self._eps.to(input_1d)
        self.window_fn = self.window_fn.to(input_1d)
        if self.config is not None and self.config["use_learnable_compression"] and torch.is_tensor(self.power):
            power = self.power.to(input_1d)
        else:
            power = self.power
        # input shape: (B,T)
        input_1d = input_1d.to(torch.float32) # to avoid float64-input
        if oneframe:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', center=False, return_complex=False).permute(0,3,2,1)
        else:
            stft_r = torch.stft(input_1d, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn,
                              pad_mode='constant', center=True, return_complex=False).permute(0,3,2,1)   # (B,2,T,F)

        if self.use_compressed_input:
            if self.config["use_learnable_compression"]:
                mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2 + self._eps) ** 0.5
                mag_spec = mag_spec.unsqueeze(1)
                phs_ = stft_r / (mag_spec + self._eps)
                mag_spec_compressed = (mag_spec + self._eps) ** power
            else:
                mag_spec = (stft_r[:,0,:,:]**2 + stft_r[:,1,:,:]**2) ** 0.5
                mag_spec = mag_spec.unsqueeze(1)
                phs_ = stft_r / torch.maximum(mag_spec, self._eps)
                mag_spec_compressed = mag_spec ** power
            in_feature = mag_spec_compressed * phs_
        else:
            in_feature = stft_r        

        return in_feature.to(torch.float32), stft_r.to(torch.float32)
    
class ISTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, window='hanning', center=True, ):
        super(ISTFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.center = center

        win_cof = scipy.signal.get_window(window, filter_length)
        self.inv_win = self.inverse_stft_window(win_cof, hop_length)

        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        inverse_basis = torch.FloatTensor(self.inv_win * \
                np.linalg.pinv(fourier_basis).T[:, None, :])

        self.register_buffer('inverse_basis', inverse_basis.float())

    def inverse_stft_window(self, window, hop_length):
        window_length = len(window)
        denom = window ** 2
        overlaps = -(-window_length // hop_length)  # Ceiling division.
        denom = np.pad(denom, (0, overlaps * hop_length - window_length), 'constant')
        denom = np.reshape(denom, (overlaps, hop_length)).sum(0)
        denom = np.tile(denom, (overlaps, 1)).reshape(overlaps * hop_length)
        return window / denom[:window_length]

    def forward(self, real_part, imag_part, length=None,oneframe=False):
        if (real_part.dim() == 2):
            real_part = real_part.unsqueeze(0)
            imag_part = imag_part.unsqueeze(0)

        recombined = torch.cat([real_part, imag_part], dim=1)
        self.inverse_basis = self.inverse_basis.to(recombined)

        inverse_transform = F.conv_transpose1d(recombined,
                                               self.inverse_basis,
                                               stride=self.hop_length,
                                               padding=0)

        padded = int(self.filter_length // 2)
        if oneframe:
            inverse_transform = inverse_transform
        else:
            if length is None:
                if self.center:
                    inverse_transform = inverse_transform[:, :, padded:-padded]
            else:
                if self.center:
                    inverse_transform = inverse_transform[:, :, padded:]
                inverse_transform = inverse_transform[:, :, :length]
        return inverse_transform
    
class Freq2TimeCodec(nn.Module):
    def __init__(self, frm_size, shift, win_len, hop_len, n_fft, config=None, power=None):
        super(Freq2TimeCodec, self).__init__()
        self.frm_size = frm_size
        self.shift = shift
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.config = config
        self._eps = torch.tensor(1e-7)
        self.istft = ISTFT(filter_length=win_len, hop_length=hop_len, window='hann',) #window='hanning', )
        if config is not None:
            self.learn_uncompressed_amp = False
            self.power = config["input_cprs_power"] if power is None else power
        else:
            self.learn_uncompressed_amp = False
        self.window_fn = torch.hann_window(win_len)

    def forward(self, dec_out, oneframe=False):
        self._eps = self._eps.to(dec_out)
        amp_output, phs_output = self._get_amp_and_phase(dec_out)                
        
        if not self.learn_uncompressed_amp:
            if self.config is not None and self.config["use_learnable_compression"]:
                power = self.power.to(dec_out)
                amp_output = (amp_output + self._eps) ** (1/(power + self._eps))
            else:
                amp_output = amp_output**(1/self.power)
        output_stft_r = amp_output * phs_output

        if (self.n_fft != self.win_len) or ((self.win_len % self.hop_len) != 0): 
            self.window_fn = self.window_fn.to(dec_out)
            if oneframe:            
                output_1d = torch.istft(output_stft_r.permute(0,3,2,1), n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn, center=False)
            else:
                output_1d = torch.istft(output_stft_r.permute(0,3,2,1), n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_len, window=self.window_fn, center=True)
        else:
            output_1d = self.istft(output_stft_r[:,0,:,:].permute(0,2,1), output_stft_r[:,1,:,:].permute(0,2,1),oneframe=oneframe)
        return output_1d.squeeze(1)

    def _get_amp_and_phase(self, x_out):
        x_out_real = x_out[:, 0, :, :]  # (B,T,F)
        x_out_imag = x_out[:, 1, :, :]
        amp = (x_out_real ** 2 + x_out_imag ** 2 + self._eps) ** 0.5
        amp = amp.unsqueeze(1) # (B,2,T,F)
        phase = x_out / (amp + self._eps)
        return amp, phase
    
class MyInstanceNorm(nn.Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, affine: bool = False, causal: bool = True):
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.affine = affine
        self.causal = causal
        self.use_unbiasEMA = True
            
        if not self.causal:
            self.norm = nn.InstanceNorm2d(channels, affine=affine)
        else:            
            # Create parameters for $\gamma$ and $\beta$ for scale and shift
            if self.affine:
                self.scale = nn.Parameter(torch.ones(channels))
                self.shift = nn.Parameter(torch.zeros(channels))
            if self.causal:
                self.pooling_kernel = 10
                self.avg = nn.AvgPool1d(kernel_size=self.pooling_kernel,stride=self.pooling_kernel)  # 100ms latency

    def forward(self, x: torch.Tensor): # B, C, T, F
        if not self.causal:
            return self.norm(x)            
        
        B,C,T,F = x.shape
        x = x.permute(0,1,3,2).reshape(B, -1, T) # B, CF, T
        #assert self.channels == x.shape[1]
        
        if not self.causal:            
            x = x.view(B, C*F, -1)
            mean = x.mean(dim=[-1], keepdim=True)
            mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
            var = mean_x2 - mean ** 2
        else:            
            x_pool_mean = self.avg(x)  # B,C,T/10
            x2_pool_mean = self.avg(x ** 2)
            _,_,T1 = x_pool_mean.shape
            mask = torch.zeros(T1, T1).to(x)  # T/10,T/10
            for i in range(1, T1 + 1):
                mask[i-1, :i] = 1
            causal_pool_nfrms = mask.sum(-1) # T/10
            
            x_pool_mean_ = x_pool_mean.unsqueeze(-2).repeat(1,1,T1,1) * mask  # B,CF,T1,T1
            mean = x_pool_mean_.sum(-1) / causal_pool_nfrms # B,CF,T1
            x2_pool_mean_ = x2_pool_mean.unsqueeze(-2).repeat(1,1,T1,1) * mask  # B,CF,T1,T1
            mean_x2 = x2_pool_mean_.sum(-1)  / causal_pool_nfrms # B,CF,T1
            var = mean_x2 - mean ** 2 # B,CF,T1
            mean = torch.repeat_interleave(mean,self.pooling_kernel,dim=-1)  # B,CF,T
            var = torch.repeat_interleave(var,self.pooling_kernel,dim=-1)  # B,CF,T
            
            if T1*self.pooling_kernel < T:
                pad_mean = torch.repeat_interleave(mean[:,:,-1].unsqueeze(-1), int(T - T1*self.pooling_kernel), dim=-1)
                mean = torch.cat((mean, pad_mean), dim=-1)
                pad_var = torch.repeat_interleave(var[:,:,-1].unsqueeze(-1), int(T - T1*self.pooling_kernel), dim=-1)
                var = torch.cat((var, pad_var), dim=-1)           
            
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C*F, -1)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.reshape(B,C,F,T).permute(0,1,3,2)
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, nonlinearity=nn.ReLU):
        super(ConvLayer, self).__init__()

        self.enc_time_kernel_len = kernel_size[0]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, bias=False)
        self.bn_conv = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()
        nn.init.xavier_normal_(self.conv.weight)

    def _pad_time(self, x, pad_size):
        """one-sided padding at time dimension for causal convolutions. Expects last two dimensions as [time x F]"""
        # https://pytorch.org/docs/stable/nn.functional.html#pad
        if pad_size:
            return F.pad(x, (pad_size[0][0], pad_size[0][1], pad_size[1][0], pad_size[1][1]))
        else:
            return F.pad(x, (0, 0, self.enc_time_kernel_len - 1, 0)) #todo

    def forward(self, x, pad_size=None):
        x = self.conv(self._pad_time(x, pad_size))
        x = self.bn_conv(x)
        return self.nonlinearity(x)

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, nonlinearity=nn.ReLU, is_last=False):
        super(DeconvLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.is_last = is_last
        if is_last:
            self.conv2d_dec_1 = ConvLayer(in_channels, out_channels, kernel_size, strides, nonlinearity= nonlinearity)
        else:
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, bias=False)
            nn.init.xavier_normal_(self.deconv.weight)   # initial
            self.bn_deconv = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
            self.nonlinearity = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

    def forward(self, deconv_x, conv_x=None, left_size=None, pad_size=None, only_current=False):
        # print('===================conv_x shape:{}'.format(conv_x.size()))
        """  """
        if self.is_last:
            deconv_out = self.conv2d_dec_1(deconv_x, pad_size=pad_size)
        else:
            # deconv block
            x = self.deconv(deconv_x)
            if left_size is not None:                
                if self.kernel_size[0] > 1:
                    if only_current:
                        x = x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, (self.kernel_size[0] - 1):-(self.kernel_size[0] - 1), left_size[0]:]
                    else:
                        x = x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :-(self.kernel_size[0] - 1), left_size[0]:]
                else:
                    x = x[:, :, :, left_size[0]:-left_size[1]] if left_size[1] else x[:, :, :, left_size[0]:]

            x = self.bn_deconv(x)
            deconv_out = self.nonlinearity(x)

        return deconv_out

class TCMLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_ker_size, rate, nonlinearity=nn.ReLU):
        super(TCMLayer, self).__init__()
        self.depth_ker_size = depth_ker_size
        self.rate = rate
 
        # conv-1
        self.conv_first = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn_first = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1)
        self.nonlinearity1 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

        # depthwise conv
        self.conv_depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=(self.depth_ker_size, 1), 
            dilation=(self.rate, 1), groups=mid_channels, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1)
        self.nonlinearity2 = nonlinearity(init=0.5) if nonlinearity == nn.PReLU else nonlinearity()

       # conv-2
        self.conv_second = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=False)

        # initial
        nn.init.xavier_normal_(self.conv_first.weight)
        nn.init.xavier_normal_(self.conv_depthwise.weight)
        nn.init.xavier_normal_(self.conv_second.weight)

    def forward(self, x, previous_frame_features=None):
        # conv-1
        x_conv_first = self.conv_first(x)
        x_conv_first = self.bn_first(x_conv_first)
        x_conv_first = self.nonlinearity1(x_conv_first)
        # depthwise conv
        if previous_frame_features is not None:
            inputs_pad = torch.cat((previous_frame_features,x_conv_first),dim=-2)
        else:
            inputs_pad = F.pad(x_conv_first, (0, 0, (self.depth_ker_size - 1)*self.rate, 0))
        x_conv_dw = self.conv_depthwise(inputs_pad)
        x_conv_dw = self.bn_depthwise(x_conv_dw)
        x_conv_dw = self.nonlinearity2(x_conv_dw)
        # conv-2
        return self.conv_second(x_conv_dw) + x, inputs_pad

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(" n: {}   overall entropy : {}".format(self.count,self.avg))

class GumbelVectorQuantizer(nn.Module):
    def __init__(self, config, input_dim, n_embeddings, groups, combine_groups, temperature=None, weight_proj_channel=-1):
        """Vector quantization using gumbel softmax
        Args:
            input_dim: input dimension (channels)
            n_embeddings: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
        """
        super().__init__()

        self.groups = 1
        self.combine_groups = False
        self.input_dim = input_dim  # 120
        self.n_embeddings = n_embeddings  # 128
        vq_dim = input_dim
        assert (
                vq_dim % self.groups == 0
        ), f"dim {vq_dim} must be divisible by groups {self.groups} for concatenation"

        embedding_dim = vq_dim // self.groups
        num_groups = self.groups if not self.combine_groups else 1

        self.embedding = nn.Parameter(torch.FloatTensor(1, num_groups * n_embeddings, embedding_dim))  # 1,1024,64
        nn.init.uniform_(self.embedding, a=-1.0, b=1.0)

        self.max_temp, self.min_temp, self.temp_decay = config["temperature"] if temperature is None else temperature
        self.curr_temp = self.max_temp
        self.codebook_indices = None
        self.config = config

        self.entropy_avg_train = AverageMeter()
        self.entropy_avg_eval = AverageMeter()
        self.code_entropy = 0
        self.tau = 0
        self.tau2 = 0.5
        self.alpha = -5
        self._eps = torch.tensor(1e-7)

    def temp_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def dequantize(self, inds, type='1'):
        ## todo: need to be checked
        ## inds [B,T]
        if type == '1':
            bsz, tsz = inds.shape
            vars = self.embedding
            if self.combine_groups:
                vars = vars.repeat(1, self.groups, 1)

            indices = inds.reshape(bsz * tsz * self.groups, -1).flatten()
            vec = torch.zeros((bsz * tsz * self.groups, self.n_embeddings)).to(vars)
            hard_x = (
                vec.scatter_(-1, indices.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
            )
            x = hard_x.view(bsz * tsz, -1)
            x = x.unsqueeze(-1) * vars
            x = x.view(bsz * tsz, self.groups, self.n_embeddings, -1)
            x = x.sum(-2)
            x = x.view(bsz, tsz, -1)
        elif type == '2':
            bsz, tsz, _ = inds.shape
            vars = self.embedding
            x = inds.unsqueeze(-1) * vars  # [BT, M, D]
            x = x.view(bsz * tsz, self.groups, self.n_embeddings, -1)
            x = x.sum(-2)  # [BT, D]
            x = x.view(bsz, tsz, -1)  # [B, T, D]
        return x

    def get_extra_losses_type2(self, target_entropy):
        mae_loss = nn.L1Loss().to(self.code_entropy)
        target_entropy = torch.tensor(1.0).to(self.code_entropy)*target_entropy
        loss = 0.01 * mae_loss(self.code_entropy, target_entropy)
        return loss

    def forward(self, x, target_entropy=0): #"max", "gumbel"
        output = {}
        _, M, D = self.embedding.size()  # 512 ,64
        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)
        self._eps = self._eps.to(x)      
        
        bsz, tsz, csz = x.shape
        x_flat = x.reshape(-1, csz)        

        distances = torch.addmm(
            torch.sum(self.embedding.squeeze(0) ** 2, dim=1) + torch.sum(x_flat ** 2, dim=1, keepdim=True), x_flat,
            self.embedding.squeeze(0).t(), alpha=-2.0, beta=1.0)  # [BT, M]
        distances_map = torch.mul(self.alpha, distances)
        distances_map = distances_map.view(bsz * tsz * self.groups, -1) # [BT, M]

        _, k = distances_map.max(-1) # [BT]
        hard_x = (distances_map.new_zeros(*distances_map.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups,-1))  # [BT, 1, M]

        hard_probs = torch.mean(hard_x.float(), dim=0)
        output["code_perplexity"] = -torch.sum(hard_probs * torch.log2(hard_probs + 1e-10), dim=-1).squeeze(0)
        avg_probs = torch.softmax(distances_map.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        output["prob_perplexity"] = -torch.sum(avg_probs * torch.log2(avg_probs + 1e-10), dim=-1).squeeze(0)
        output["temp"] = self.curr_temp

        if self.training:
            # print(x.view(bsz * tsz * self.groups, -1).argmax(dim=-1))
            distances_map = F.gumbel_softmax(distances_map.float(), tau=self.curr_temp, hard=True).type_as(distances_map)
            # Compute entropy loss
            self.code_entropy = output["prob_perplexity"]  # rate control (entropy for current batch)
            # Weighted entropy regularization term
            weighted_code_entropy[:, 0] = self.code_entropy
            weighted_code_entropy[:, 1] = self.tau
            # overall entropy (for current epoch)
            self.entropy_avg_train.update(self.code_entropy.detach())
            avg_entropy = self.entropy_avg_train.avg
            # if avg_entropy < 1:
            #     self.max_temp = min(1.005*self.max_temp, 5.0)
        else:
            distances_map = hard_x
            self.code_entropy = output["code_perplexity"]
            # Weighted entropy regularization term
            weighted_code_entropy[:, 0] = self.code_entropy
            weighted_code_entropy[:, 1] = self.tau
            # overall entropy
            self.entropy_avg_eval.update(self.code_entropy.detach())
            avg_entropy = self.entropy_avg_eval.avg

        distances_map = distances_map.view(bsz * tsz, -1)
        vars = self.embedding  # [1, M, D]
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        output["quantization_inds"] = (
            distances_map.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()
        )       

        distances_map = distances_map.unsqueeze(-1) * vars  # [BT, M, D]
        distances_map = distances_map.view(bsz * tsz, self.groups, self.n_embeddings, -1)
        distances_map = distances_map.sum(-2)  # [BT, D]
        quantized = distances_map.view(bsz, tsz, -1) # [B, T, D]

        output["quantized_feature"] = quantized
        output["entropy"] = self.code_entropy # (entropy for current batch)
        output["entropy_avg"] = avg_entropy #  (entropy for current epoch)
        if self.config["Train_cfg"]["use_entropy_loss"]:
            output["entropy_loss"] = self.get_extra_losses_type2(target_entropy)
 
        output["commitment_loss"] = F.mse_loss(x, quantized.detach()) ## just for log
        return output
        
class MultiFrmVQBottleNeck(nn.Module):
    def __init__(self, feat_dim, bitrate, sampling_rate=16000, config=None):
        super(MultiFrmVQBottleNeck, self).__init__()

        self.bitrate_dict = {'0.256k':256,'0.512k':512,'1k':1000,'2k':2000,'3k':3000,'6k':6000,'9k':9000,'10k':10000,'12k':12000,'24k':24000}
        self.bitrate = bitrate     
        self.config = config            
        self.feat_dim = feat_dim 
        self.combine_frames = config["combineVQ_frames"] 

        self.latent_dim = config['latent_dim']
        self.codebook_num = config['codebook_num']
        self.codebook_size = [config['codebook_size'] for i in range(self.codebook_num)]  # codeword number
        self.target_entropy = config['target_entropy']
        
        self.codebook_dim = self.latent_dim * self.combine_frames // self.codebook_num
        self.conv1x1_1 = nn.Conv2d(self.feat_dim, self.latent_dim, kernel_size=1, stride=1,bias=False)
        nn.init.xavier_normal_(self.conv1x1_1.weight)
        self.conv1x1_2 = nn.Conv2d(self.latent_dim, self.feat_dim, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_normal_(self.conv1x1_2.weight)        

        self.vq_layer_list = nn.ModuleList(
                GumbelVectorQuantizer(config, input_dim=self.codebook_dim, n_embeddings=i, groups=1, combine_groups=False) for i in self.codebook_size)      

    def forward(self, inputs, loss_mask=None, epo=None): # (B, C, T, 1)
        return

    def vq_bottleneck(self, inputs, bitrate=None, loss_mask=None, epo=None): # (B, C, T, 1)
        result = {}
                   
        enc_feat = inputs   
        
        self.bitrate = self.config['bitrate']
        
        bitrate = self.bitrate
        result["bitrate"] = bitrate
        # input shape: (B, L)           
                 
        batch_size, nb_frames = enc_feat.shape[0], enc_feat.shape[2]
        enc_feat = self.conv1x1_1(enc_feat)  # [B,120,T,1]
        enc_feat_r = enc_feat.squeeze(-1).permute(0, 2, 1) #[B,T,C]
        
        vq_out = self.quantize(enc_feat_r, bitrate=bitrate, epo=epo)
        
        vq_feat_before_1x1 = vq_out["vq_feat"] # [B, T, C] unfolded
        vq_feat = vq_out["vq_feat"].permute(0, 2, 1).unsqueeze(-1)  # [B,320,T,1]               
        vq_feat = self.conv1x1_2(vq_feat)  # [B,320,T,1]

        vq_out_keys = ["quantization_inds", "prob_perplexity_list", "code_perplexity_list", "entropy_loss", "entropy", "entropy_avg"]
        for key in vq_out_keys:        
            if key in vq_out.keys():
                result.update({key: vq_out[key]}) 

        result["vq_feat"] = vq_feat  
        result["feat_before_vq"] = enc_feat_r 
        result["feat_after_vq"] = vq_feat_before_1x1     
        return result  

    def quantize(self, enc_feat, bitrate=None, epo=None):
        ### vector quantization
        #input shape [B,T,C]
        result = {}
        vq_layer_out = []
        vq_inds_out = []
        prob_perplexity_list = []
        code_perplexity_list = []
        entropy_list = []
        entropy_avg_list = []
        entropy_loss = 0
        if self.config["Train_cfg"]["use_entropy_loss"]:
            target_entropy_per_vqlayer = torch.tensor(self.target_entropy/self.codebook_num).to(enc_feat)     
        
        B, T, channels = enc_feat.shape # B,T,C'
        combine_frames = self.combine_frames           
        enc_feat_combine = enc_feat.reshape(B, T// combine_frames, combine_frames, channels) # B,T/2,2,C'                    

        enc_feat_combine = torch.split(enc_feat_combine, channels//self.codebook_num,dim=-1)  # codebook_num,B,T/2,2,C'//codebook_num
        valid_vq_list = self.vq_layer_list 
        for layer_i in range(len(valid_vq_list)):
            vq_layer = self.vq_layer_list[layer_i]
            if self.combine_frames > 1:
                vq_in = enc_feat_combine[layer_i].reshape(B, T // combine_frames, -1)
            else:
                vq_in = enc_feat[:, :, layer_i*self.codebook_dim:(layer_i+1)*self.codebook_dim]
            #self.vq_in_list.append(vq_in.reshape(B*T // combine_frames,-1)) # just for visual
            if self.config["Train_cfg"]["use_entropy_loss"]:
                vq_out = vq_layer(vq_in, target_entropy_per_vqlayer)
            else:
                vq_out = vq_layer(vq_in)
            if self.combine_frames > 1:
                vq_layer_out.append(vq_out["quantized_feature"].reshape(B, T//combine_frames, combine_frames, channels//self.codebook_num))
            else:
                vq_layer_out.append(vq_out["quantized_feature"])
            vq_inds_out.append(vq_out["quantization_inds"]) 
            if 'entropy' in vq_out.keys(): # softmax prob
                entropy_list.append(vq_out["entropy"]) # entropy for current batch
                entropy_avg_list.append(vq_out["entropy_avg"]) # entropy for current epoch
            if 'prob_perplexity' in vq_out.keys(): # soft prob
                prob_perplexity_list.append(vq_out["prob_perplexity"])
            if 'code_perplexity' in vq_out.keys(): # hard prob
                code_perplexity_list.append(vq_out["code_perplexity"])
            if 'entropy_loss' in vq_out.keys():
                entropy_loss += vq_out["entropy_loss"]                
                
        if self.combine_frames > 1:
            vq_feat = torch.cat(vq_layer_out, dim=-1) # B,T/2,2,C'                    
            result["vq_feat"] = vq_feat.reshape(B,T,channels)
        else:
            result["vq_feat"] =  torch.cat(vq_layer_out, dim=-1)  # B,T,320
        
        result["quantization_inds"] =  torch.cat(vq_inds_out,dim=-1)  # B,T,codebook_num
        if 'prob_perplexity' in vq_out.keys(): # soft prob
            result["prob_perplexity_list"] = prob_perplexity_list  # [codebook_num]
        if 'code_perplexity' in vq_out.keys(): # hard prob
            result["code_perplexity_list"] = code_perplexity_list  # [codebook_num]
        if 'entropy_loss' in vq_out.keys():
            result["entropy_loss"] = entropy_loss #
        if 'entropy' in vq_out.keys():
            result["entropy"] = entropy_list  #
            result["entropy_avg"] = entropy_avg_list  #          

        return result
        
    def dequantize(self, vq_inds):
        vq_layer_out = []
        if self.combine_frames > 1:
            B, T = vq_inds.shape[0:2] # B,T,C'

        for layer_i in range(len(self.vq_layer_list)):
            vq_layer = self.vq_layer_list[layer_i]
            if len(vq_inds.shape) == 3:
                vq_out = vq_layer.dequantize(vq_inds[:,:,layer_i],type='1')
            elif len(vq_inds.shape) == 4:
                vq_out = vq_layer.dequantize(vq_inds[:,:,layer_i,:],type='2')
            #vq_out = vq_layer.dequantize(vq_inds[:,:,layer_i])
            if self.combine_frames > 1:
                vq_layer_out.append(vq_out.reshape(B, T, self.combine_frames,-1))  # [B,T/2,2,C'//codebook_num]
            else:
                vq_layer_out.append(vq_out)  # [B,T,C'//codebook_num]
        vq_feat = torch.cat(vq_layer_out, dim=-1)  # B,T,C'
        if self.combine_frames > 1:
            vq_feat = vq_feat.reshape(B, T*self.combine_frames, -1)
        return vq_feat
    
    def decode_vq_bottleneck(self, vq_inds):
        batch_size, nb_frames = vq_inds.shape[0], vq_inds.shape[1]
        vq_feat = self.dequantize(vq_inds)      # [B, T, C]  
        vq_feat = vq_feat.permute(0, 2, 1).unsqueeze(-1)  # [B,320,T,1]                       
        vq_feat = self.conv1x1_2(vq_feat)  # [B,320,T,1]   
        return vq_feat

    def reset_entropy_hists_train(self):
        for vq_layer in self.vq_layer_list:
            vq_layer.entropy_avg_train.reset()

    def reset_entropy_hists_eval(self):
        for vq_layer in self.vq_layer_list:
            vq_layer.entropy_avg_eval.reset()

    def get_overall_entropy_avg_train(self, bitrate=' '):      
        avgs = []
        for vq_layer in self.vq_layer_list:
            avgs.append(vq_layer.entropy_avg_train.avg)
        return [torch.stack(avgs, dim=0).sum(dim=0)]

    def get_overall_entropy_avg_eval(self):
        avgs = []
        for vq_layer in self.vq_layer_list:
            avgs.append(vq_layer.entropy_avg_eval.avg)
        return [torch.stack(avgs, dim=0).sum(dim=0)]
        
    def update_temperature_gumbel(self, cur_iter):    
        for vq_layer in self.vq_layer_list:
            vq_layer.temp_updates(cur_iter)

def rpad_signal_for_codec(signal, combine_frames, hop_len):
    # pad for center stft
    signal_len = signal.shape[-1]
    pad_len = 0
    if signal_len % hop_len:
        pad_len = (int(signal_len / hop_len) + 1)*hop_len - signal_len        
    
    # pad for combine_frames   
    num_frames = int((signal_len + pad_len) / hop_len) + 1    
    if num_frames % combine_frames:
        padded_frames = combine_frames - num_frames % combine_frames
        pad_len += int(padded_frames * hop_len) 
        
    if pad_len > 0 and (signal is not None):
        signal = torch.nn.functional.pad(signal, (0, pad_len))
        
    return signal, pad_len

    
    