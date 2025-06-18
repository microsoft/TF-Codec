# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import argparse   
import yaml
import librosa
import soundfile as sf    

from models.tfcodec import TFCodec

def audioread(path):        
    '''Function to read audio'''
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=0, stop=None)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')
    if len(audio.shape) > 1:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0)/audio.shape[0]
    return audio, sample_rate

def audiowrite(destpath, audio, sample_rate=16000):
    '''Function to write audio'''
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path')
    parser.add_argument('--model_path')
    parser.add_argument('--config_path', default='configs/tfcodec_config_6k.yaml')
    parser.add_argument('--output_path')
    return parser.parse_args()

if __name__ == "__main__": 
    args = parser()
    config = {}
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f:
        config_vq = yaml.safe_load(f)
        if "model_cfg" in config.keys():
            vq_cfg = config_vq['Model_cfg'][config["model_cfg"]]
            config.update(vq_cfg)
        common_cfg = config_vq['Common_cfg']
        config.update(common_cfg)

    model = TFCodec(config=config).cuda(f'cuda:{0}') 
    model.load(args.model_path)
    model.eval()

    input_wav, sr = audioread(args.audio_path)
    if(sr != config['sampling_rate']):
        input_wav = librosa.resample(input_wav, orig_sr=sr, target_sr=config['sampling_rate'])            
    input_length = len(input_wav)
    result = model(torch.tensor(np.expand_dims(input_wav, axis=0)).cuda(f'cuda:{0}').to(torch.float))
    bitrate = config["bitrate"]
    output_rec = result[bitrate]["x_hat"]                  
    output_wav = torch.reshape(torch.Tensor.cpu(output_rec), (-1,)).cpu().detach().numpy()
    audiowrite(args.output_path, output_wav, sample_rate=config['sampling_rate']) 

    # # to get quantized feature only
    # result = model.encode_to_token_idx(torch.tensor(np.expand_dims(input_wav, axis=0)).cuda(f'cuda:{0}').to(torch.float))
    # decoded_feature = result["vq_feat"].squeeze(-1) # [B, C, T] at 5ms resolution for 6kbps