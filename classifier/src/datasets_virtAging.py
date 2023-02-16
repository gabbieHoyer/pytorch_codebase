# -*- coding: utf-8 -*-
import os
import shutil
from os.path import exists
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import time
import torchvision
from torchvision import transforms
from utils import *
from shutil import copyfile
# %% codecell
import torch
import h5py
import pickle
import pdb
from itertools import combinations
from typing import List, Union
import torch.nn.functional as F
import torch
from tqdm import tqdm
from torch import Tensor, is_tensor
from scans import *
from torch.utils.data import Dataset
import pickle

class OAClassification(Dataset):
    def __init__(self, bones: List[str], split: str, limit=float('inf'), noise: float=0, num_channels_model=3,
                 group='CartilageThickness'):
        assert split in ['train', 'val', 'test']
        assert all([bone in ['Femur', 'Tibia', 'Patella'] for bone in bones])
        assert not(any([a == b for a, b in combinations(bones, 2)]))
        self.noise = noise
        self.dataset = list()
        self.limit = limit
        self.group = group
        self.num_channels_model=num_channels_model
        f = open('/data/VirtualAging/DataLookup/kMRI_data.pickle', 'rb')
        data = pickle.load(f)

        all_scans = data['scans']
        f.close()
        g = open('/data/VirtualAging/DataLookup/classification_splits.pickle', 'rb')
        scan_ids = pickle.load(g)[split]
        for bone in tqdm(bones):
            for scan_id, KL in tqdm(scan_ids):
                scan: Scan = all_scans[scan_id]
                img: SphericalMaps = getattr(scan, bone)['SphericalMaps']
                path: str = getattr(img, self.group)
		if Path(path).is_dir():
                    continue		
                scratch_path: Path = self.h5_to_pt(Path(path))
                self.dataset.append({'img': scratch_path, 'oa': int(KL >= 2)})
        g.close()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, int]:
        sample = self.dataset[round(idx % self.limit)]
        img = load_sample(sample['img'], self.num_channels_model)
        return img, sample['oa']

    def h5_to_pt(self, path: Path) -> Path:
        pt_path: Path = Path(str(path).split('.')[0] + '.pt')
        scratch_path: Path = Path(str(pt_path).replace('data', '/data/VirtualAging/users/ghoyer/Virtual_Aging_Cart_Thickness/scratch', 1))
        if not scratch_path.is_file():
            scratch_path.parent.mkdir(parents=True, exist_ok=True)
            if pt_path.is_file():
                shutil.copy(pt_path, scratch_path)
            else:
                with h5py.File(path, 'r') as f:
                    torched_img = torch.tensor(f[self.group])
                torch.save(torched_img, pt_path)
                torch.save(torched_img, scratch_path)
        return pt_path


def gaussian_noise(image: Tensor, mean: int = 0, var: float = 0.01) -> Tensor:
    sigma = var ** 0.5
    return image + (sigma * torch.randn(image.shape) + mean)


def load_sample(path: str, num_channels_model=1) -> Tensor:

    image = torch.load(path)
    image /= 5
    return image.repeat(num_channels_model, 1, 1)


def get_dataloaders(args):
    """ Gets the dataloaders for the chosen dataset.
    """    
    if not isinstance(args.bone, list):
        args.bone= [args.bone]
    if args.dataset in ['CartilageThickness']:
        group = args.dataset.split('_')[-1]
        datasets = {phase: OAClassification(bones=args.bone, split=phase, num_channels_model=args.n_chans,
                                            group=group) for phase in ['train', 'val', 'test']}
        dataloaders = {phase: DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle= phase == 'train', num_workers=0) for phase, dataset in datasets.items()}
       
        args.class_names = ('normal', 'abnormal')  # 0,1 labels
    else:
        NotImplementedError('{} dataset not available.'.format(args.dataset))

    return dataloaders, args
