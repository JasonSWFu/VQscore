#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import soundfile as sf
from torch.utils.data import Dataset

class SingleDataset(Dataset):
    def __init__(
        self,
        data_path,
        files,
        query="*.wav",
        load_fn=sf.read,
        return_utt_id=False,
        subset_num=-1,
        batch_length=9600
    ):
        self.return_utt_id = return_utt_id
        self.load_fn = load_fn
        self.subset_num = subset_num
        self.data_path = data_path
        self.batch_length = batch_length
        self.filenames = self._load_list(files, query)
        # self.filenames = pd.read_csv(files)['Filaname']
        self.utt_ids = self._load_ids(self.filenames)


    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        data = self._data(idx)
        
        if self.return_utt_id:
            items = utt_id, data
        else:
            items = data
            

        return items


    def __len__(self):
        return len(self.filenames)
    

    def _load_list(self, files, query):
#         if isinstance(files, list):
#             filenames = files
#         else:
#             if os.path.exists(files):
#                 filenames = sorted(find_files(files, query))
#             else:
#                 raise ValueError(f"{files} is not a list or a existing folder!")
            
#         if self.subset_num > 0:
#             filenames = filenames[:self.subset_num]
#         assert len(filenames) != 0, f"File list in empty!"
        # return filenames
        filenames = pd.read_csv(files)['Filename'].to_list()
        filenames = [os.path.join(self.data_path,filename) for filename in filenames]
        return filenames
    
    
    def _load_ids(self, filenames):
        utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in filenames
        ]
        return utt_ids
    

    def _data(self, idx):
        return self._load_data(self.filenames[idx], self.load_fn)
    

    def _load_data(self, filename, load_fn):
        # Join the root directory with the relative file path
        filename = os.path.join(self.data_path, filename.lstrip('/'))
        
        if load_fn == sf.read:            
            data = load_fn(filename, always_2d=True)[0][:,0] # T x C, 1
            data_shape = data.shape[0]
            if data.shape[0]<=self.batch_length:
                data = np.concatenate((data, np.zeros(self.batch_length-data.shape[0])))[None,:].astype(np.float32)
            else:
                start = np.random.randint(0,(data.shape[0]-self.batch_length+1))
                data = data[None, start : start + self.batch_length].astype(np.float32)
        else:
            data = load_fn(filename)
        return data, data_shape


