# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
from utils.utils import str2bool

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_path')
    parser.add_argument('--checkpoint_path') 
    parser.add_argument('--train_data_dir', help='Trainging data.')
    parser.add_argument('--val_data_dir', default=None, help='Validation data.')
    parser.add_argument('--train_dir', default=None, help='Path to save training results.')
    parser.add_argument('--config', default="config.yaml")
    parser.add_argument('--nproc_per_node')
    parser.add_argument('--nnodes')
    parser.add_argument(
        "--use_ddp_launch",
        type=str2bool,
        default=False,
        help="use_ddp_launch",
    )
    parser.add_argument('--num_workers')
    return parser.parse_args()

if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    args = parser()

    entry_cmd = "python3 -m torch.distributed.launch" #"torchrun --standalone" #"python3 -m torch.distributed.launch"
    nproc_per_node = "--nproc_per_node={}".format(args.nproc_per_node)
    nnodes = "--nnodes={}".format(args.nnodes)
    num_workers = "--num_workers={}".format(args.num_workers)
    entry_py = "train.py"
    train_data_dir = "--train_data_dir={}".format(args.train_data_dir)

    entry_cmd = entry_cmd + ' ' + nproc_per_node + ' ' + nnodes + ' ' + entry_py + ' ' + train_data_dir + ' ' + num_workers

    if args.use_ddp_launch is not None:
        use_ddp_launch = "--use_ddp_launch={}".format(args.use_ddp_launch)
        entry_cmd += ' ' + use_ddp_launch

    if args.train_dir is not None:
        train_dir = "--train_dir={}".format(args.train_dir)
        entry_cmd = entry_cmd + ' ' + train_dir

    if args.val_data_dir is not None:
        val_data_dir = "--val_data_dir={}".format(args.val_data_dir)
        entry_cmd = entry_cmd + ' ' + val_data_dir

    if args.warmup_path is not None:
        warmup_path = "--warmup_path={}".format(args.warmup_path)
        entry_cmd = entry_cmd + ' ' + warmup_path

    if args.checkpoint_path is not None:
        checkpoint_path = "--checkpoint_path={}".format(args.checkpoint_path)
        entry_cmd = entry_cmd + ' ' + checkpoint_path

    if args.config is not None:
        train_config = "--config={}".format(args.config) 
        entry_cmd = entry_cmd + ' ' + train_config

    os.system(entry_cmd)

    
    