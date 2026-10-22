from torch.utils.data import Dataset
import torch
import random
import numpy as np
import Encoding

class WP2TensorDatasetV4(Dataset):
    def __init__(self, path: str, length: int, data: str, dtype=torch.bfloat16, mode='training', seed=0):
        self.data = data
        self.dtype = dtype
        self.path = path
        self.mode = mode
        self.seed = seed
        self.length = length

        self.tensor = torch.load(self.path, mmap=True, weights_only=True)

        self.context_options = {
            '1kb': 1024, '2kb': 2048, '4kb': 4096, '8kb': 8192,
            '16kb': 16384, '32kb': 32768, '64kb': 65536,
            '128kb': 131072, '512kb': 524288, '200kb': 262074,
            '300kb': 327680, '524kb': 524294, '544kb': 557056
        }

        self.rc = [0, 1]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        sample = self.tensor[index]

        if self.data == 'training':
            rc_choice = np.random.choice(self.rc)

            if rc_choice == 0:
                batch_cpg = sample[0:1, :].to(dtype=self.dtype)
                batch_5mC = sample[1:2, :].to(dtype=self.dtype)
                batch_m6A = sample[2:3, :].to(dtype=self.dtype)
                batch_state = sample[3:4, :].to(dtype=self.dtype)
            else:
                batch_cpg = torch.flip(sample[0:1, :], dims=(-1,)).to(dtype=self.dtype)
                batch_5mC = torch.flip(sample[1:2, :], dims=(-1,)).to(dtype=self.dtype)
                batch_m6A = torch.flip(sample[2:3, :], dims=(-1,)).to(dtype=self.dtype)
                batch_state = torch.flip(sample[3:4, :], dims=(-1,)).to(dtype=self.dtype)

            return batch_cpg, batch_5mC, batch_m6A, batch_state[:, 512]

        elif self.data == 'validation':
            batch_cpg = sample[0:1, :].to(dtype=self.dtype)
            batch_5mC = sample[1:2, :].to(dtype=self.dtype)
            batch_m6A = sample[2:3, :].to(dtype=self.dtype)
            batch_state = sample[3:4, :].to(dtype=self.dtype)

            return batch_cpg, batch_5mC, batch_m6A, batch_state[:, 512]