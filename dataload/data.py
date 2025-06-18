# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pyarrow as pa
import lmdb
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class LmdbData(Dataset):
    def __init__(self, db_path, config=None, split='train'):
        self.db_path = db_path
        self.sr = config['sampling_rate']
        self.config =  config
        self.split = split
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        #return {"id": index, "target_signal": torch.tensor(unpacked[self.config["Train_cfg"]['target_data_idx']], dtype=torch.float32)}                
        return unpacked            

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def collater(self, samples):
        ## online augmentation
        audios = [s["target_signal"] for s in samples]

        batch = {
            "target_signal": torch.stack(audios,0),
        }
        return batch
       
class DistributedDataLoader():
    def __init__(self,config=None) -> None:
        self.config = config
        pass

    def worker_init_fn(self, worker_id):
        np.random.seed()
        random.seed()
        seed = np.random.get_state()[1][0] + worker_id
        print("worker_id, seed:", worker_id, seed)
        print("rand numbers:", np.random.randint(0, 10, 5))

    def get_dataloader(self, split, data_dir, batch_size, num_workers, prefetch_factor, ddp=True):        
        if data_dir is None:
            return None, None
        
        dataset = LmdbData(data_dir, self.config, split=split)

        if split == 'train':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if ddp else None
            shuffle = True if (sampler is None) else False
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, 
                collate_fn=None, pin_memory=True, prefetch_factor=prefetch_factor,
                sampler=sampler, drop_last=True, worker_init_fn=self.worker_init_fn, shuffle=shuffle, batch_sampler=None)
        else:  # 'eval' or 'dev'
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if ddp else None
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                collate_fn=None, pin_memory=True, prefetch_factor=prefetch_factor,
                sampler=sampler, worker_init_fn=self.worker_init_fn, batch_sampler=None)
        return data_loader, sampler
    