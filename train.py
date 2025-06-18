# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import time
import datetime
import yaml
import shutil
import random
import math
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn.utils.rnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from azureml.core.run import Run

from model_caller import ModelCaller
from loss_caller import LossCaller
from dataload.data import DistributedDataLoader
from log import get_logger
from optim.optimizer import Optimizer
from utils.utils import *

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt2 = rt / nprocs
    return rt2.detach().cpu()

def model_size(model):
    trainable_num, total_num = 0, 0
    for name, module in model.named_children():
        for k_name, parameter in module.named_parameters():
            total_num += parameter.numel()
            if parameter.requires_grad:
                trainable_num += parameter.numel()  ## h0 in rnn not included?

    print('Total params:{}  Trainable: {}'.format(total_num, trainable_num))
    print('Model size:{:.2f}MB  '.format(float(total_num) * 4 / 1024 / 1024))

def set_optimizers(config, model_caller):
    lr_scheduler = None
    if not config["new_opt"]:
        optimizer = torch.optim.Adam(model_caller.model.parameters(), lr=config["learning_rate"])
        #optimizer = torch.optim.Adam(model_caller.model.parameters(), lr=config["learning_rate"])
        if config["use_warmup"]:
            lr_lambda = lambda step: min(1, step / 6000.0)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        opt_list = optimizer
        assert not config["use_adv_loss"]
    else:
        optimizer = Optimizer(config)
        optimizer.set_optimizer(config["optimizer"]["opt_G"], model_para=(model_caller.model.parameters()))
        optimizer.set_scheduler(None)
        opt_list = [optimizer]

        if config["use_adv_loss"]:
            my_disc = model_caller.discriminator.module if model_caller.is_ddp else model_caller.discriminator
            opt_d_cur = Optimizer(config)
            opt_d_cur.set_optimizer(config["optimizer"]["opt_D"], model_para=(my_disc.discriminator.parameters()))
            opt_d_cur.set_scheduler(None)
            opt_list.append(opt_d_cur)        
    return opt_list, lr_scheduler

def parse_model_filename(model_name):
    model_name = model_name.split('.ckpt')[0]
    loss = float(model_name.split('loss-')[-1].split('-')[0]) if 'loss' in model_name else float(0)
    iter = float(model_name.split('iter-')[-1].split('-')[0]) if 'loss' in model_name else float(0)
    return loss, iter

def check_ckpt_num(directory):
    max_keep = config.get('max_keep', 20)
    ckpt_pths = glob.glob(f'{directory}/*.ckpt')
    if len(ckpt_pths) >= max_keep:
        # ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
        err_list,iter_list,acc_list,ckpt_list = [],[],[],[]
        for model in ckpt_pths:
            model_name = os.path.basename(model)
            loss, iter = parse_model_filename(model)
            err_list.append(loss)
            iter_list.append(iter)
            ckpt_list.append(model)
        df_val = pd.DataFrame({'iter': iter_list, 'err': err_list, 'ckpt': ckpt_list, })
        score = 0
        _loss_rank,_iter_rank = 1,0
        if _loss_rank: #rank model according to loss
            err_rank = df_val["err"].rank(ascending=True,na_option='bottom')
            score += err_rank
        if _iter_rank: #rank model according to iteration
            iter_rank = df_val["iter"].rank(ascending=False)
            score += iter_rank
        topn_index = sorted(range(len(score)), key=lambda i: score[i],reverse=True)
        for i in range(len(ckpt_pths) - max_keep):
            model_path = df_val['ckpt'][topn_index[i]]
            os.remove(model_path)

def main_worker(local_rank, rank, world_size, num_workers, config, train_logger, use_ddp_launch=False):
    print("Use rank {} local rank {} for training".format(rank, local_rank))
    setup_dist(local_rank, rank, world_size, use_ddp_launch=use_ddp_launch) 

    input_data_idx = config["Train_cfg"]["input_data_idx"]
    target_data_idx = config["Train_cfg"]["target_data_idx"]
    model_caller = ModelCaller(config, train_logger.logger)
    model_size(model_caller.model)  

    model_caller.distribute_model(True, rank=local_rank, )
    batch_size, val_batch_size = model_caller.batch_size, model_caller.val_batch_size
  
    loss_caller = LossCaller(config["Train_cfg"]["loss_type"], config, local_rank) 
    opt_list, lr_scheduler = set_optimizers(config["Train_cfg"], model_caller)

    if config["warmup_path"]:
        opt_list = model_caller.restore_model(config["warmup_path"], global_params=True, optimizer=opt_list)
    elif config["checkpoint_path"] is not None:
        model_caller.restore_model(config["checkpoint_path"], global_params=False, )   
    else:
        model_caller.restore_model()        
        
    strips = int(time.time())
    net_opt_filepath = os.path.join(config["train_dir"], "warmup")
    ckpt_filepaths = glob.glob(r"{}/model_*.ckpt".format(net_opt_filepath))
    if len(ckpt_filepaths) > 0 and (config["warmup_path"] is None):
        net_opt_filepath = ckpt_filepaths[-1]
        opt_list = model_caller.restore_model(net_opt_filepath, global_params=True, optimizer=opt_list)
    else:
        os.makedirs(net_opt_filepath, exist_ok=True)
        net_opt_filepath = os.path.join(net_opt_filepath, "model_{}.ckpt".format(strips))       

    train_logger.logger.info("%s training with data: %s", config["model_type"], config["train_data_dir"])
    prefetch_factor = batch_size if torch.cuda.is_available() else None ## prefetch_factor need to be None when debugging with cpu
    val_prefetch_factor = val_batch_size if torch.cuda.is_available() else None

    dataloader_caller = DistributedDataLoader(config)
    train_loader, train_sampler = dataloader_caller.get_dataloader('train', config['train_data_dir'], batch_size, num_workers, prefetch_factor, model_caller.is_ddp)    
    data_loader_val, val_sampler = dataloader_caller.get_dataloader('eval', config['val_data_dir'], val_batch_size, num_workers, val_prefetch_factor, model_caller.is_ddp)
      
    ### train and validate
    train_logger.logger.info('Start training!')
    start_epoch = model_caller.start_epoch
    for epoch in range(start_epoch, config["Train_cfg"]["num_epochs"]):
        if not config["Train_cfg"]['new_opt']:
            if lr_scheduler is not None:
                train_logger.logger.info('Current learning rate {}'.format(lr_scheduler.get_last_lr()))
        else:
            train_logger.logger.info('Current learning rate {}'.format(opt_list[0].get_lr()))

        if model_caller.is_ddp:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        train(net_opt_filepath, train_loader, data_loader_val, model_caller, loss_caller, opt_list, epoch, local_rank, rank, world_size, config, train_logger, lr_scheduler=lr_scheduler)
        
        # save the model and optimizer for warmup
        if rank==0:
            model_caller.save_model(net_opt_filepath, True, epoch, opt_list)
                
        if opt_d_sd is not None:
            opt_d_sd.scheduler.step()

    dist.barrier()
    cleanup_dist()

def train(net_opt_filepath, data_loader, data_loader_val, model_caller, loss_caller, optimizer, epoch, local_rank, rank, world_size, config, train_logger, lr_scheduler=None):
    total_step = len(data_loader)
    tot_iter_base = total_step * epoch

    train_cfg = config["Train_cfg"]
    runavg_gradnorm = train_cfg["max_grad_norm"]
    input_data_idx, target_data_idx = train_cfg["input_data_idx"], train_cfg["target_data_idx"]
        
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = model_caller.model.module if model_caller.is_ddp else model_caller.model
    opt_g = optimizer[0] if train_cfg['new_opt'] else optimizer
    data_loader_iter = iter(data_loader)

    try:
        ii = 0
        while True:
            seq = next(data_loader_iter)
            cur_iter = tot_iter_base + ii
            # Forward pass
            input_batch = seq if isinstance(seq, torch.Tensor) else seq[input_data_idx]
            target_sig = seq.to(device, non_blocking=True) if isinstance(seq, torch.Tensor) else seq[target_data_idx].to(device, non_blocking=True)
            if train_cfg["random_crop_duration"] > 0:
                audio_len = input_batch.size()[-1]
                crop_len = math.floor(train_cfg["random_crop_duration"] * config['sampling_rate'])
                start = max(math.floor(torch.rand(1)* (audio_len - crop_len)), 0)
                input_batch = input_batch[:, start:start+crop_len]
                target_sig = target_sig[:, start:start+crop_len]
            pred_sig = model_caller.call_model(input_batch, split='train', epoch=epoch)            
            loss = loss_caller.compute_loss(pred_sig, target_sig, None, model)

            total_norm = {'gen': torch.tensor(0).to(device)}            
            cur_bitrate = config['bitrate']
            loss_g = loss['total_loss']

            my_disc = None
            if train_cfg["use_adv_loss"]:
                my_disc = model_caller.discriminator.module if model_caller.is_ddp else model_caller.discriminator                      
                total_norm.update({my_disc.disc_name: torch.tensor(0).to(device)})
                loss_d, loss_d_real, loss_d_fake = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device)
                loss_g_adv, loss_g_feat = torch.tensor(0.).to(device), torch.tensor(0.).to(device)

                ## update discriminator
                if cur_iter % my_disc.disc_cfg["n_generator"] == 0:  ## train disc every n_generator                    
                    pred = pred_sig[cur_bitrate]["x_hat"].detach()                
                    dis_fake, dis_real, _, _ = model_caller.call_discriminator(pred, target_sig, cur_bitrate, 'train')
                    loss_d_cur, loss_d_real_cur, loss_d_fake_cur = loss_caller.adversarial_loss_dis(dis_fake, dis_real)
                    loss_d_real, loss_d_fake = loss_d_real_cur * my_disc.dist_w, loss_d_fake_cur * my_disc.dist_w
                    loss_d = loss_d_cur * my_disc.dist_w                    
                    optimizer[1].zero_grad()
                    loss_d.backward()
                    total_norm[my_disc.disc_name] = optimizer[1].step(my_disc.discriminator.parameters(), cur_iter)        

                ## update generator every iteration after warmup
                if my_disc.disc_cfg["D_warmup_time"] <= cur_iter:
                    pred = pred_sig[cur_bitrate]["x_hat"]
                    dis_fake, dis_real, dis_fake_feat_cur, dis_real_feat_cur = model_caller.call_discriminator(pred, target_sig, cur_bitrate, 'train')
                    loss_g_adv_cur = loss_caller.adversarial_loss_gen(dis_fake)
                    loss_g_feat_cur = loss_caller.compute_featloss_on_disc(dis_fake_feat_cur, dis_real_feat_cur)
                    loss_g_adv, loss_g_feat = loss_g_adv_cur * my_disc.dist_w_g, loss_g_feat_cur * my_disc.dist_w_g           
                  
                loss_g += my_disc.disc_cfg["adv_loss_weight"] * loss_g_adv
                loss_g += my_disc.disc_cfg["disc_feat_loss_weight"] * loss_g_feat
      
            if (not train_cfg["use_adv_loss"]) or (train_cfg["use_adv_loss"] and not my_disc.disc_cfg["freeze_gen"]):
                if config["Train_cfg"]['new_opt']:
                    opt_g.zero_grad()
                    loss_g.backward()
                    para_g = model_caller.model.parameters()
                    total_norm_g = opt_g.step(para_g, cur_iter)
                    total_norm['gen'] = total_norm_g
                else:
                    opt_g.zero_grad()
                    loss_g.backward()
                    model_para = model_caller.model.parameters()
                    if train_cfg['disable_adaptive_grad_norm']:
                        for para in model_para:
                            torch.nn.utils.clip_grad_norm_(para, runavg_gradnorm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(model_para, 1.5 * runavg_gradnorm)
                        runavg_gradnorm = 0.999 * runavg_gradnorm + 0.001 * total_norm.item()

                    opt_g.step()
                    
                    # hard clip power_cprs
                    if config["use_learnable_compression"]:
                        for name, para in model.named_parameters():
                            if 'power_cprs' in name:
                                para.data = torch.clamp(para.data, min=0.2, max=1.0)
                            
                    if lr_scheduler is not None:
                        lr_scheduler.step()
            
            # Compute the average, training log
            if cur_iter % train_cfg["log_step"] == 0:
                dist.barrier()
                bitrate = config['bitrate']
                
                reduced_loss = {}                
                for k, v in loss[bitrate].items():
                    if k not in reduced_loss.keys():
                        reduced_loss[k] = []
                    reduced_loss[k].append(reduce_mean(v, world_size))

                reduced_loss_adv = {}
                if train_cfg["use_adv_loss"]:    
                    adv_loss_k_list = ["G_total", "G_adv", "D_adv", "D_real", "D_fake", "G_feat_loss"]
                    adv_loss_v_list = [loss_g, loss_g_adv, loss_d, loss_d_real, loss_d_fake, loss_g_feat]
                    for k, v in zip(adv_loss_k_list, adv_loss_v_list):
                        reduced_loss_adv.update({k: [reduce_mean(v, world_size)]})

                if train_cfg["use_entropy_loss"]:
                    entropy_avg = model.get_overall_entropy_avg_train()
                    if isinstance(entropy_avg, torch.Tensor):
                        reduced_entropy = [reduce_mean(entropy_avg, world_size)]
                    else:  # list
                        reduced_entropy = []
                        for entropy_item in entropy_avg:
                            curr_reduced_entropy = reduce_mean(entropy_item.to(device), world_size)
                            reduced_entropy.append(curr_reduced_entropy)

                ### Print log for current step
                if rank == 0:
                    errdict = {}
                    for k, v in reduced_loss.items():
                        errdict[k] = v  # list
                    for k, v in reduced_loss_adv.items():
                        errdict[k] = v
                    if train_cfg["use_entropy_loss"]:
                        errdict['entropy_avg_train'] = reduced_entropy
                    train_logger.refresh_train_state_vqvae(epoch, total_step, ii, errdict)

            # validation
            if not (train_cfg["sav_per_num_iter"]==-1) and (((cur_iter+1) % train_cfg["sav_per_num_iter"] == 0) or ((cur_iter+1) % train_cfg["log_val_step"] == 0)):
                validate(data_loader_val, model_caller, loss_caller, local_rank, rank, world_size, cur_iter, config, train_logger=train_logger)
                
                if not (train_cfg["sav_global_per_num_iter"]==-1) and ((cur_iter+1) % train_cfg["sav_global_per_num_iter"] == 0):                
                    if rank == 0:
                        model_caller.save_model(net_opt_filepath, True, epoch, optimizer)
  
            if train_cfg["use_entropy_loss"]:
                model.update_temperature_gumbel(cur_iter)
            ii += 1
    except StopIteration:
        if train_cfg["use_entropy_loss"]:
            model.reset_entropy_hists_train()  # reset entropy every epoch
        if rank == 0: ## save model every epoch
            save_dir = os.path.join(os.path.dirname(train_logger.path),'epoch_model')
            os.makedirs(save_dir,exist_ok=True)
            train_logger.logger.info("Saving model to: %s", os.path.join(save_dir,os.path.basename(train_logger.path)))
            model_caller.save_model(os.path.join(save_dir, os.path.basename(train_logger.path)))
            check_ckpt_num(config["train_dir"])

def validate(data_loader, model_caller, loss_caller, local_rank, rank, world_size, iter, config, train_logger=None):
    input_data_idx, target_data_idx = config["Train_cfg"]["input_data_idx"], config["Train_cfg"]["target_data_idx"]
    train_cfg = config["Train_cfg"]

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = model_caller.model.module if model_caller.is_ddp else model_caller.model
     
    loss = {}         
    
    batch_idx = 0
    with torch.no_grad():        
        for seq_batch in data_loader:
            input_batch = seq_batch if isinstance(seq_batch, torch.Tensor) else seq_batch[input_data_idx]  
            target_batch = seq_batch.cuda(local_rank, non_blocking=True) if isinstance(seq_batch, torch.Tensor) else seq_batch[target_data_idx].cuda(local_rank, non_blocking=True)
            if train_cfg["val_crop_duration"] > 0:              
                crop_len = math.floor(train_cfg["val_crop_duration"] * config['sampling_rate'])
                input_batch = input_batch[:, 0:crop_len]
                target_batch = target_batch[:, 0:crop_len]                
            pred_batch = model_caller.call_model(input_batch, split='eval', )               
            out_criterion = loss_caller.compute_loss(pred_batch, target_batch, None, model)
            
            bitrate_list = [config['bitrate']]
            for i, bitrate in enumerate(bitrate_list):                    
                for k, v in out_criterion[bitrate].items():
                    if 'val_'+k not in loss.keys():
                        loss['val_'+k] = [AverageMeter() for i in range(len(bitrate_list))] 
                    loss['val_'+k][i].update(out_criterion[bitrate][k])                                  

            batch_idx += 1
            
        cprs_pow_dict = []
        if train_cfg["use_entropy_loss"]:                      
            entropy_avg = model.get_overall_entropy_avg_eval()
            if isinstance(entropy_avg, torch.Tensor):
                reduced_entropy = [reduce_mean(entropy_avg, world_size)]
            else:  # list
                reduced_entropy = []
                for entropy_item in entropy_avg:
                    reduced_entropy.append(reduce_mean(entropy_item, world_size))
            model.reset_entropy_hists_eval()
            
        if config["use_learnable_compression"]:
            for name, para in model.named_parameters(): 
                if 'power_cprs' in name:                        
                    cprs_pow_dict.append(para.data)

        if batch_idx != 0:
            dist.barrier()
            reduced_loss = {}               
            for i, bitrate in enumerate(bitrate_list):
                for k, v in loss.items():
                    if k not in reduced_loss.keys():
                        reduced_loss[k] = [] 
                    reduced_loss[k].append(reduce_mean(loss[k][i].avg, world_size))                    

            if rank == 0:
                errdict = {}
                for k, v in reduced_loss.items():
                    errdict[k] = v
                if train_cfg["use_entropy_loss"]:
                    errdict.update({"val_entropy_avg": reduced_entropy}) 
                if config["use_learnable_compression"]:
                    errdict.update({"val_cprs_power": cprs_pow_dict})
            
                train_logger.refresh_eval_state_vqvae(reduced_loss["val_loss"], iter, errdict)                        
                    
        if rank == 0 and ((iter+1) % train_cfg["sav_per_num_iter"] == 0):        
            if not os.path.isdir(config["train_dir"]):
                os.makedirs(config["train_dir"])
            # save the model weights
            train_logger.logger.info("Saving model to: %s", train_logger.path)
            model_caller.save_model(train_logger.path)            

            os.makedirs(os.path.join(config["train_dir"],'latest_iter'),exist_ok=True)
            model_caller.save_model(os.path.join(os.path.join(config["train_dir"],'latest_iter'), 'latest_iter.ckpt'))

class Logger:
    def __init__(self, config) -> None:
        self.run = Run.get_context()
        self.logger = get_logger(__name__)

        self.tags = {}
        if config["Train_cfg"]["tags"] is not None:
            self.tags.update(config["Train_cfg"]["tags"])

        current_time = datetime.datetime.today().strftime("%Y_%m_%d-%H_%M_%S")
        self.logger.log_to_file(os.path.join(config["train_dir"], "./logs/l-" + current_time + ".log"))
        self.logger.info("Training. tags: %s", self.tags)
        self.tensorboard_writer = SummaryWriter(os.path.join(config["train_dir"], "tensorboard"))

        self.total_epoch = config["Train_cfg"]['num_epochs']
        self.model_type = config["model_type"]
        self.train_dir = config["train_dir"]

        self.config = config
        
    def refresh_eval_state_vqvae(self, reduced_loss, cur_iter, reduced_loss_dict):
        for key, value in reduced_loss_dict.items():
            for i, v in enumerate(value):
                self.run.log('val/{}_{}'.format(key, i), v.item())
            
        sum_mse_loss = np.sum(np.asarray(reduced_loss_dict['val_mse_loss'])) if 'val_mse_loss' in reduced_loss_dict.keys() else 0
        sum_mel_loss = np.sum(np.asarray(reduced_loss_dict['val_mel_loss'])) if 'val_mel_loss' in reduced_loss_dict.keys() else 0
        sum_total_rec_loss = sum_mse_loss + self.config["Train_cfg"]["mel_loss_weight"]*sum_mel_loss
        entropy_avg = np.sum(np.asarray(reduced_loss_dict['val_entropy_avg'])) if 'val_entropy_avg' in reduced_loss_dict.keys() else 0        
        sum_total_loss = np.sum(np.asarray(reduced_loss))
            
        errstr = "{}: {:2.5f}, {}: {:2.5f}, {}: {:2.5f}".format('val_loss', sum_total_loss, 'val_mse', sum_mse_loss, 'val_mel', sum_mel_loss, 'val_entropy_avg', entropy_avg)
        self.logger.info("Validation iter %3d, %s", cur_iter + 1, errstr)
        
        sum_vq_loss = np.sum(np.asarray(reduced_loss_dict['val_vq_loss'])) if "val_vq_loss" in reduced_loss_dict.keys() else 0

        for key, value in reduced_loss_dict.items():
            for i, v in enumerate(value):
                self.tensorboard_writer.add_scalar('val/{}_{}'.format(key, i), v.item(), cur_iter + 1)

        savstr = "totalrecon-{:.6f}-loss-{:.6f}".format(sum_total_rec_loss, sum_total_loss)
        if "val_mel_loss" in reduced_loss_dict.keys():
            savstr = "{}-mel-{:.6f}".format(savstr, sum_mel_loss)
        if "val_mse_loss" in reduced_loss_dict.keys():
            savstr = "{}-mse-{:.6f}".format(savstr, sum_mse_loss)
        if 'val_entropy_avg' in reduced_loss_dict.keys():
            savstr = "{}-entropy-{:.6f}".format(savstr, entropy_avg)
        if 'val_vq_loss' in reduced_loss_dict.keys():
            savstr = "{}-vq-{:.6f}".format(savstr, sum_vq_loss)
        fname = "{}-{}-{}-{}.ckpt".format(self.model_type, savstr, "iter", cur_iter + 1)
        fpath = os.path.join(self.train_dir, fname)
        self.path = fpath
        self.tags.update({"iter": cur_iter, "error": sum_total_loss,}) 
       
    def log_message(self, msg):
        self.logger.info("Msg %s", msg)
                
    def refresh_train_state_vqvae(self, epoch, total_step, ii, reduced_loss_dict):
        for key, value in reduced_loss_dict.items():
            for i, v in enumerate(value):
                self.run.log('train/{}_{}'.format(key, i), v.item())
            
        sum_mse_loss = np.sum(np.asarray(reduced_loss_dict['mse_loss'])) if 'mse_loss' in reduced_loss_dict.keys() else 0
        sum_mel_loss = np.sum(np.asarray(reduced_loss_dict['mel_loss'])) if 'mel_loss' in reduced_loss_dict.keys() else 0
        entropy_avg = np.sum(np.asarray(reduced_loss_dict['entropy_avg_train'])) if 'entropy_avg_train' in reduced_loss_dict.keys() else 0
        errstr = "{}: {:2.5f}, {}: {:2.5f}, {}: {:2.5f}, {}: {:3.2f}".format("loss", np.sum(np.asarray(reduced_loss_dict['loss'])), "mse", sum_mse_loss, "mel", sum_mel_loss, "entropy_avg_train", entropy_avg)
        self.logger.info("Epoch [%3d/%3d], Step [%3d/%3d], %s", epoch + 1, self.total_epoch, ii + 1, total_step, errstr)

        tot_iter_base = total_step * epoch
        for key, value in reduced_loss_dict.items():
            for i, v in enumerate(value):
                self.tensorboard_writer.add_scalar('train/{}_{}'.format(key, i), v.item(), tot_iter_base + ii + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and validate TF-Codec models.')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--nproc_per_node', type=int)
    parser.add_argument('--nnodes', type=int)
    parser.add_argument(
        "--use_ddp_launch",
        type=str2bool,
        default=False,
        help="use_ddp_launch",
    )
    parser.add_argument('--num_workers', type=int, default=4)    
    parser.add_argument('--warmup_path')    
    parser.add_argument('--checkpoint_path') 
    parser.add_argument('--train_data_dir', help='Training data.')
    parser.add_argument('--val_data_dir', default=None, help='Validation data.')
    parser.add_argument('--train_dir', default=None, help='Path to save training results.')
    parser.add_argument('--config', default="config.yaml")
    args = parser.parse_args()

    config = {}
    print("Using config from ", args.config)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    with open("configs/model_config.yaml") as f:
        config_vq = yaml.safe_load(f)
        if "model_cfg" in config.keys():
            vq_cfg = config_vq['Model_cfg'][config["model_cfg"]]
            config.update(vq_cfg)
        common_cfg = config_vq['Common_cfg']
        config.update(common_cfg)

    config["train_data_dir"] = args.train_data_dir
    config["val_data_dir"] = args.val_data_dir
    config["checkpoint_path"] = args.checkpoint_path
    config["warmup_path"] = args.warmup_path
    config["train_dir"] = args.train_dir    
    print("Proceeding with config", config)  

    os.makedirs(config['train_dir'], exist_ok=True)
    shutil.copy(args.config, config['train_dir'])
    time.sleep(1.0)

    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        assert args.nnodes is not None and args.nproc_per_node is not None, "nnodes and nproc_per_node should be set manually."
        world_size = args.nnodes * args.nproc_per_node
    assert world_size >= 1
    print(f"world size is {world_size}")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["NCCL_ASYNC_ERROR_HANDING"] = "1"    

    train_logger = Logger(config)
    main_worker(local_rank, rank, world_size, args.num_workers, config, train_logger, use_ddp_launch=args.use_ddp_launch)


