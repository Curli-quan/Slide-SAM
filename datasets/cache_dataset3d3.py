"""
    re-index by masks! not images
"""

import numpy as np
import os
from einops import rearrange, reduce, repeat
from tutils.nn.data import read, itk_to_np, np_to_itk, write
from tutils import tfilename
from .cache_dataset3d import Dataset3D as basic_3d_dataset
from monai import transforms
import torch
import cv2
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange
import glob
from torchvision import transforms
from monai import transforms as monai_transforms


class Dataset3D(basic_3d_dataset):
    def __init__(self, config=..., use_cache=True, *args, **kwargs) -> None:
        super().__init__(config, use_cache=use_cache, *args, **kwargs)
        self.basic_dir = config['data_root_path']
        self.cache_dir = config['cache_data_path']
    
    def prepare_cached_datalist(self):
        config = self.config        
        data_paths = []
        for dirpath in glob.glob(config['cache_data_path'] + "/*"):
            if not os.path.isdir(dirpath):
                continue
            prefix = dirpath.split("/")[-1]
            if prefix.split("_")[0] in config['cache_prefix']:
                data_paths += glob.glob(dirpath + "/label_jpg/*.jpg")
                print("Load ", dirpath)
        print('Masks len {}'.format(len(data_paths)))
        print('Examples: ', data_paths[:2])
        return data_paths  
    
    def _get_cached_data(self, index):
        mask_path = self.img_names[index]
        # print(name)
        mask = np.int32(cv2.imread(mask_path) > 0)

        prefix = mask_path[:-9]
        img_path = prefix.replace("/label_jpg/label_", "/image/image_") + ".jpg"
        img = cv2.imread(img_path)
        number = int(mask_path[-8:-4])
        
        meta = np.load(prefix.replace("/label_jpg/label_", "/meta/meta_")+".npy", allow_pickle=True).tolist()
        # label_idx = np.random.randint(0, label_ori.shape[0])
        
        return rearrange(img, "h w c -> c h w"), rearrange(mask, "h w c -> c h w"), mask_path, meta['labels'][number], meta['label_idx'][number]
    
    # @tfunctime
    def __getitem__(self, index, debug=False):
        index = index % len(self)
        img_rgb, label_ori, name, label_idx, local_idx = self._get_cached_data(index)

        # assert label_ori.sum() > 0
        if label_ori.sum() <= 0:
            print("[Label Error] ", name)
            return self.__getitem__(index+1)

        # assert len(img_rgb.shape) == 3, f"{__file__} Got{img_rgb.shape}"
        # img_rgb = self.transform((img_rgb[None,:,:,:]))
        img_rgb = F.interpolate(torch.Tensor(img_rgb).unsqueeze(0), size=(1024,1024)).squeeze().numpy()
        
        vector = np.ones(3)
        ret_dict = {
            "name": name,
            "img": img_rgb,
            "label": label_ori,
            "indicators": vector,
            "class": label_idx,
            "local_idx": local_idx,
            "is_problem": label_ori.sum() <= 30,
        } 
        return ret_dict

if __name__ == "__main__":
    # def go_cache():
    from tutils.new.manager import ConfigManager
    from tqdm import tqdm
    config  = ConfigManager()
    config.add_config("configs/vit_b.yaml")
    config.print()
    dataset = Dataset3D(config=config['dataset'], use_cache=True)
    # dataset.caching_data()
    # dataset.convert_masks_types()
    for i in tqdm(range(len(dataset))):
        data = dataset.__getitem__(i)
        # assert 
    
