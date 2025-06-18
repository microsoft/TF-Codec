# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from losses.base_loss import MSE, MAE
from losses.power_compressed_mse import PowerCompressedMSE
from losses.multiscale_mel_loss import MultiscaleMelLoss

class LossCaller(object):
    def __init__(self, loss_type, config, rank):
        if torch.cuda.is_available():
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")

        self.config = config
        self.train_cfg = config["Train_cfg"]
        self.device = device

        TYPE_CHOICE = ("power-compressed-mse",)
        if loss_type not in TYPE_CHOICE:
            raise ValueError("Invalid loss type: " + loss_type)
        
        self.quality_loss = PowerCompressedMSE(config, rank)
        self.multiscaleMelloss = MultiscaleMelLoss(config, config["sampling_rate"], rank)

        if self.train_cfg["use_adv_loss"]:
            self.disc_cfg = self.train_cfg["GAN_cfg"]
            if self.disc_cfg["adversarial_loss"] == 'MSE':
                self.mseloss = MSE(config, rank)
            self.feat_loss = MAE(config, rank)        

    def compute_loss(self, output, target_1d, aux_target_1d=None, model=None):
        cost = {}
        total_cost = 0

        bitrate_list = [self.config['bitrate']]

        for bitrate in bitrate_list:
            cost[bitrate] = {}
            cost[bitrate]["loss"] = 0
            if self.train_cfg["use_bin_loss"] and ("x_hat" in output[bitrate].keys()):
                cost[bitrate]["mse_loss"] = self.quality_loss.compute_loss(output[bitrate]["x_hat"], target_1d, self.config['sampling_rate'])
                cost[bitrate]["loss"] += self.train_cfg["bin_loss_weight"] * cost[bitrate]["mse_loss"]

            if self.train_cfg["use_multiscale_mel_loss"] and ("x_hat" in output[bitrate].keys()):
                cost[bitrate]["mel_loss"] = self.multiscaleMelloss.compute_loss(output[bitrate]["x_hat"], target_1d)
                cost[bitrate]["loss"] += self.train_cfg["mel_loss_weight"] * cost[bitrate]["mel_loss"]
                
            if "commitment_loss" in output[bitrate].keys():
                cost[bitrate]["vq_loss"] = output[bitrate]["commitment_loss"]
                if self.train_cfg["use_vq_loss"]:
                    cost[bitrate]["loss"] += self.train_cfg["vq_loss_weight"] * cost[bitrate]["vq_loss"]

            if "entropy_loss" in output[bitrate].keys():
                cost[bitrate]["entropy_loss"] = output[bitrate]["entropy_loss"]
                if self.train_cfg["use_entropy_loss"]:
                    cost[bitrate]["loss"] += self.train_cfg["entropy_loss_weight"] * cost[bitrate]["entropy_loss"]                                                                                             

            total_cost += cost[bitrate]["loss"]             

        cost["total_loss"] = total_cost
        return cost         

    def adversarial_loss_gen(self, dis_fake):
        if self.disc_cfg["adversarial_loss"] == 'hinge':
            loss_g = torch.mean(F.relu(1. - dis_fake))
            # loss_g = -torch.mean(dis_fake)
        elif self.disc_cfg["adversarial_loss"] == 'MSE':
            valid = torch.ones_like(dis_fake).to(dis_fake)
            loss_g = self.mseloss.compute_loss(dis_fake, valid)
        elif self.disc_cfg["adversarial_loss"] == 'BCE':
            valid = torch.ones_like(dis_fake).to(dis_fake)
            loss_g = self.bceloss(dis_fake, valid)

        return loss_g

    def adversarial_loss_dis(self, dis_fake, dis_real):
        if self.disc_cfg["adversarial_loss"] == 'hinge':
            loss_real = torch.mean(F.relu(1. - dis_real))
            loss_fake = torch.mean(F.relu(1. + dis_fake))
        elif self.disc_cfg["adversarial_loss"] == 'MSE':
            valid = torch.ones_like(dis_real).to(dis_real)
            fake = torch.zeros_like(dis_fake).to(dis_fake)
            loss_real = self.mseloss.compute_loss(dis_real, valid)
            loss_fake = self.mseloss.compute_loss(dis_fake, fake)
        elif self.disc_cfg["adversarial_loss"] == 'BCE':
            valid = torch.ones_like(dis_real).to(dis_real)
            fake = torch.zeros_like(dis_fake).to(dis_fake)
            loss_real = self.bceloss(dis_real, valid)
            loss_fake = self.bceloss(dis_fake, fake)
        loss_d =  loss_real + loss_fake
        return loss_d,loss_real,loss_fake

    def compute_featloss_on_disc(self, pred_feat_list, gt_feat_list):
        loss = torch.tensor(0.).to(self.device)
        for pred_feat,gt_feat in zip(pred_feat_list, gt_feat_list):
            loss_i = self.feat_loss.compute_loss(pred_feat, gt_feat)
            loss += loss_i
        loss /= len(pred_feat_list)
        return loss