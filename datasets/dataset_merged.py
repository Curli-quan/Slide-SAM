# from torchvision import transforms
from monai import transforms
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import torch.nn.functional as F
import glob
import os
from einops import rearrange

from tutils.nn.data import read, itk_to_np

from tqdm import tqdm
from tutils import tfilename, tdir
import random
# from .dataset2d import Dataset2D
from .dataset3d_2dmask import Dataset2D
from .dataset3d import Dataset3D


class DatasetMerged(dataset):
    def __init__(self, config=None, is_train=True, getting_multi_mask=False) -> None:
        super().__init__()
        self.dataset2d = Dataset2D(dirpath="/quanquan/datasets/08_AbdomenCT-1K/", is_train=True)
        self.dataset3d = Dataset3D(config=config, is_train=True)
        self.len_2d = len(self.dataset2d)
        self.len_3d = len(self.dataset3d)

    def __getitem__(self, index, debug=False):
        index = index % len(self)
        # print("DEBUG! is_2d:", index < self.len_2d)
        if index < self.len_2d:
            return self.dataset2d.__getitem__(index)
        else:
            index = (index - self.len_2d) % self.len_3d
            return self.dataset3d.__getitem__(index)

    def __len__(self):        
        return len(self.dataset2d) + len(self.dataset3d) * 200
    


class TestsetMerged(dataset):
    def __init__(self, config=None, is_train=False) -> None:
        super().__init__()
        self.dataset2d = Dataset2D(dirpath="/quanquan/datasets/08_AbdomenCT-1K/preprocessed/", is_train=False)
        self.dataset3d = Dataset3D(config=config, is_train=False, split='val')
        self.len_2d = len(self.dataset2d)
        self.len_3d = len(self.dataset3d)

    def __getitem__(self, index, debug=False):
        index = index % len(self)
        if index < self.len_2d:
            return self.dataset2d.__getitem__(index)
        else:
            index = (index - self.len_2d) % self.len_3d
            return self.dataset3d.__getitem__(index)

    def __len__(self):        
        return len(self.dataset2d) + len(self.dataset3d) * 2
    
    
if __name__ == "__main__":
    from tutils import timer
    from tutils.new.manager import trans_args, trans_init, ConfigManager
    config = ConfigManager()
    config.add_basic_config()
    config.add_config("configs/vit_b.yaml")
    dataset = DatasetMerged(config['dataset'])
    tt = timer()
    for i in range(20000,len(dataset)):
        data = dataset.__getitem__(i)
        print("time: ", tt())