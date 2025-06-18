# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import shutil
import os
import torch
import argparse

import datetime  
from torch import distributed as dist

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def setup_dist(
    local_rank, rank, world_size, master_port=None, use_ddp_launch=False, master_addr=None
):
    """
    rank and world_size are used only if use_ddp_launch is False.
    """

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )
   
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)
  
    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(864000))
        torch.cuda.set_device(local_rank)
    else:
        dist.init_process_group("nccl", timeout=datetime.timedelta(864000))


def cleanup_dist():
    dist.destroy_process_group()

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