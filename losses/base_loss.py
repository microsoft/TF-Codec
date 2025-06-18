# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


class MSE(object):
    def __init__(self, config, rank=0) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        self.mse_loss = nn.MSELoss().cuda(device)

    def compute_loss(self, pred, target, ):
        loss = self.mse_loss(pred, target)
        return loss


class MAE(object):
    def __init__(self, config, rank=0) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
        self.mae_loss = nn.L1Loss().cuda(device)

    def compute_loss(self, pred, target, ):
        loss = self.mae_loss(pred, target)
        return loss
