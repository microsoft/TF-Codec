# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import torch
import random

import torch.distributed as dist   
from models.tfcodec import TFCodec  
    
class ModelCaller():
    def __init__(self, config, logger):
        self.batch_size = config["Train_cfg"]['batch_size']
        self.val_batch_size = config["Train_cfg"]['val_batch_size']
        self.model_type = config["model_type"]
        self.config = config
        self.GAN_cfg = config["Train_cfg"]["GAN_cfg"]
        self.logger = logger
        self.device = None     
        
        self.model = TFCodec(config=config)       
        if self.config["Train_cfg"]["use_adv_loss"]:
            from models.adversarial_modules.discriminator import Discriminator
            self.discriminator = Discriminator(config)          

    def distribute_model(self, is_ddp, rank=None, ):
        self.is_ddp = is_ddp
        find_unused_params = True
        if not torch.cuda.is_available():
            device = torch.device("cpu")
            self.is_ddp = False
            print('Using CPU, this will be slow')
        elif is_ddp:
            if rank is not None:                                
                torch.cuda.set_device(rank)                
                device = torch.device("cuda", rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda(rank)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_params)
                if self.config["Train_cfg"]["use_adv_loss"]:
                    self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator).cuda(rank)
                    self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_params)
            else:
                device = torch.device("cuda")
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=find_unused_params)
                if self.config["Train_cfg"]["use_adv_loss"]:
                    self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator).cuda()
                    self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator, find_unused_parameters=find_unused_params)
        elif rank is not None:
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
            self.model.cuda(rank)
            if self.config["Train_cfg"]["use_adv_loss"]:
                self.discriminator.cuda(rank)
        else:
            device = torch.device("cuda")
            self.model = self.model.cuda()
            if self.config["Train_cfg"]["use_adv_loss"]:
                self.discriminator.cuda()

        self.device = device

    def _restore_model(self, model_dict, type='generator'):
        new_dict = {}

        for key in model_dict.keys():
            if type == 'generator':
                new_key = key.split('generator.')[-1]  ## to compatible with older gan-versions
            if type == 'discriminator':
                new_key = key.split('discriminator.')[-1]  ## to compatible with older gan-versions

            # print(new_key)
            module = new_key.split('module.')[-1]
            module = module.split('.')[0] if '.' in module else module
            # print(module)

            if self.is_ddp:  # load non-ddp model or ddp model to ddp model
                new_key = 'module.' + new_key if 'module' not in new_key else new_key
            else:  # load non-ddp model or ddp model to non-ddp model
                new_key = new_key.split('module.')[-1]
            if type == 'generator':
                if 'discriminator' not in key: ## to compatible with older gan-versions
                    new_dict[new_key] = model_dict[key]
            if type == 'discriminator':
                if 'generator' not in key: ## to compatible with older gan-versions
                    new_dict[new_key] = model_dict[key]
        return new_dict        

    def restore_model(self, ckpt_path=None, global_params=False, optimizer=None):
        if ckpt_path is not None:
            if global_params and (optimizer is not None) and (ckpt_path is not None):  # load parameters and optimizer
                if self.logger is not None:
                    self.logger.info("Warm start from [%s].", ckpt_path)
                tmp_dict = torch.load(ckpt_path, map_location=self.device)
                self.start_epoch = tmp_dict['epoch'] + 1
                ## load generator
                gen_dict = tmp_dict["gen"] if 'gen' in tmp_dict.keys() else tmp_dict["net"]
                new_model_dict = self.model.state_dict()
                new_dict = self._restore_model(gen_dict, 'generator')
                new_dict_opt = {k: v for k, v in new_dict.items() if k in new_model_dict}
                new_model_dict.update(new_dict_opt)
                self.model.load_state_dict(new_model_dict)
                if self.config['Train_cfg']['new_opt']:
                    optimizer[0].optimizer.load_state_dict(tmp_dict['optimizer'])  ## list[0]??
                    for state in optimizer[0].optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                else:
                    optimizer.load_state_dict(tmp_dict['optimizer'])
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                if self.config["Train_cfg"]["use_adv_loss"]:
                    my_disc = self.discriminator.module if self.is_ddp else self.discriminator
                    my_disc.restore(self, self.device, tmp_dict, optimizer=optimizer)
            
            else:  # load parameters only
                if self.logger is not None:
                    self.logger.info("Load model from [%s].", ckpt_path)

                if self.is_ddp:
                    self.model.module.load(ckpt_path)
                else:
                    self.model.load(ckpt_path)
                self.start_epoch = 0                                    
                           
                if self.config["Train_cfg"]["use_adv_loss"] and self.GAN_cfg["load_disc_para"]:
                    my_disc = self.discriminator.module if self.is_ddp else self.discriminator
                    my_disc.restore(self, self.device, tmp_dict, optimizer=None)
        else:
            if self.logger is not None:
                self.logger.info('Start from scratch')
            self.start_epoch = 0

        return optimizer                                 

    def save_model(self, ckpt_path, global_params=False, epoch=0, optimizer=None, ):
        if global_params and (optimizer is not None):
            state = {}     
            if isinstance(optimizer, list):
                state = {'epoch': epoch, 'gen': self.model.state_dict(),'optimizer': optimizer[0].optimizer.state_dict()}
                if self.config["Train_cfg"]["use_adv_loss"]:
                    my_disc = self.discriminator.module if self.is_ddp else self.discriminator
                    state.update(my_disc.my_state_dict(optimizer=optimizer))
            else:
                state = {'epoch': epoch, 'gen': self.model.state_dict(),'optimizer': optimizer.state_dict()}
            torch.save(state, ckpt_path)
        else:
            if self.config["Train_cfg"]["use_adv_loss"]:
                dict = {}
                my_disc = self.discriminator.module if self.is_ddp else self.discriminator
                dict.update(my_disc.my_state_dict(optimizer=None))
                dict.update({'gen': self.model.state_dict()})
                torch.save(dict, ckpt_path)
            else:
                torch.save(self.model.state_dict(), ckpt_path) 

    def call_discriminator(self, signal, target, bitrate=None, split='train'):
        my_disc = self.discriminator.module if self.is_ddp else self.discriminator
        if split == 'train':
            my_disc.train()
        else:  # 'eval'
            my_disc.eval()

        return my_disc(signal.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True))      

    def call_model(self, seq_input, split, epoch=None, iter=None, bitrate=None):
        if split == 'train':
            self.model.train()
        else: 
            self.model.eval()
        
        input = seq_input.to(self.device, non_blocking=True)
        return self.model(input)

