# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

class Optimizer(object):
    def __init__(self, config, optim=None, scheduler=None):
        self.optimizer = optim
        self.scheduler = scheduler
        self.config = config

    def step(self, model_para, num_iter):
        ## clip gradient
        norm_list = []
        for para in model_para:
            norm_list.append(torch.nn.utils.clip_grad_norm_(para, self.config["max_grad_norm"]))
        norm = torch.mean(torch.Tensor(norm_list))
        self.optimizer.step()

        if self.scheduler is not None:
            self.update(num_iter)

        return norm

    def set_optimizer(self, optim_config, model_para):
        optim_cfg = {"name":'adam', "lr":1e-3, "betas":(0.9, 0.999), "weight_decay":0} ## default setting
        for key, value in optim_config.items():
            optim_cfg[key] = value
        if optim_config["name"] == 'adam':
            self.optimizer = torch.optim.Adam(model_para,weight_decay=optim_cfg["weight_decay"],betas=optim_cfg["betas"])
            self.set_lr(optim_cfg["lr"]) ## lr initial

    def set_scheduler(self, scheduler):
        if scheduler == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[35,], gamma=0.05)
        elif scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.97)
        else:
            self.scheduler = None

    def update(self,num_iter):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass
        elif self.scheduler is None:
            pass
        else:
            self.scheduler.step(num_iter)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr