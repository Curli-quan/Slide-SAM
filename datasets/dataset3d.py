from torchvision import transforms
from monai import transforms as monai_transforms
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
# from monai.transforms import SpatialPadd, CenterSpatialCropd, Resized, NormalizeIntensityd
# from monai.transforms import RandAdjustContrastd, RandShiftIntensityd, Rotated, RandAffined
# from datasets.common_2d_aug import RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from tutils import tfilename, tdir
import random
import time
import cv2
from scipy.sparse import csr_matrix

def tfunctime(func):
    def run(*argv, **kargs):
        t1 = time.time()
        ret = func(*argv, **kargs)
        t2 = time.time()
        print(f"[Function {func.__name__}] Running time:{(t2-t1):.6f}s")
        return ret
    return run

# DEFAULT_PATH='/home1/quanquan/datasets/KiTS/'
DEFAULT_PATH="/home1/quanquan/datasets/BCV-Abdomen/Training/"

LABEL_INDICES={
            "t2sag": ["bg","kidney", "label 2", "label 3", "rectum", "tumor", "other"],
        }

# CACHE_DISK_DIR="/home1/quanquan/code/projects/medical-guangdong/cache/data2d_3/"
CACHE_DISK_DIR=None
# DEFAULT_CONFIG={
#     "pad": (512,512),
#     "crop": (384,384),
#     "resize": (512,512),
# }

DATASET_CONFIG={
    'split': 'train',
    'data_root_path':'/quanquan/datasets/',
    'dataset_list': ['sam', "their", "ours"],
    'data_txt_path':'./datasets/dataset_list/',
}

DATASET_METAINFO={
    "WORD": {0:"Background", 1:"Liver", 2:"Spleen", 3:"Left Kidney", 4:"Right Kidney", 5:"Stomach", 6:"Gallbladder", 7:"Esophagus", 8:"Pancreas", 9:"Duodenum", 10:"Colon", 11:"Intestine", 12:"Adrenal", 13:"Rectum", 14:"Bladder", 15:"left head of femur", 16:"right head of femur"}
}


class Dataset3D(dataset):
    def __init__(self, config=DATASET_CONFIG, is_train=True, split='train', getting_multi_mask=False, use_cache=False) -> None:
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.split = split
        self.getting_multi_mask = getting_multi_mask
        self.use_cache = use_cache
        self.img_names = self.prepare_cached_datalist() if use_cache else self.prepare_datalist() 
        # self.img_names = self.prepare_datalist()
        self.prepare_transforms()
    
    def prepare_cached_datalist(self):
        raise NotImplementedError

    def prepare_transforms(self):
        self.transform = monai_transforms.Compose([
            # transforms.Resized(keys=['img', 'label'], spatial_size=(3,512,512)),
            # transforms.RandSpatialCropd(keys=["img", 'label'], roi_size=(3,448,448)),
            # transforms.RandAffined(keys=['img', 'label'], prob=0.5, shear_range=(0.2,0.2)),
            # transforms.RandCropByPosNegLabeld(keys=['img', 'label'], spatial_size=(3,448,448), label_key='label', neg=0),
            # transforms.RandSmoothFieldAdjustContrastd(keys=['img', 'label'], )
            monai_transforms.RandAdjustContrastd(keys=['img'], ),
            # transforms.RandShiftIntensityd(keys=['img'], prob=0.8, offsets=(0, 20)),
            monai_transforms.Resized(keys=['img', 'label'], spatial_size=(3,1024,1024)),
        ])
        self.test_transform = transforms.Compose([
            monai_transforms.Resized(keys=['img', 'label'], spatial_size=(3,1024,1024)),            
        ])
    
    def _get_image(self, index):
        name = self.img_names[index]['img_path']
        if not os.path.exists(name):
            print("Path not exists!", name)
            return self._get_image(index+1%len(self))
        img_itk = read(self.img_names[index]['img_path'])
        img_ori = itk_to_np(img_itk)
        img = np.clip(img_ori, -200, 400).astype(np.float32)
        img = (img - img.min()) / img.max() * 255
        label_ori = itk_to_np(read(self.img_names[index]['label_path']))
        return {"img":img, "name":name.replace(self.config['data_root_path'], ""), "label":label_ori}
        
    def __len__(self):
        return len(self.img_names)
    
    # @tfunctime
    def prepare_datalist(self):
        config = self.config
        data_paths = []
        for item in config['dataset_list']:
            print("Load datalist from ", item)
            for line in open(config["data_txt_path"]+ item + f"_{self.split}.txt"):
                name = line.strip().split()[1].split('.')[0]
                img_path = config['data_root_path'] + line.strip().split()[0]
                label_path = config['data_root_path'] + line.strip().split()[1]
                data_paths.append({'img_path': img_path, 'label_path': label_path, 'name': name})
        print('train len {}'.format(len(data_paths)))
        return data_paths
    
    # @tfunctime
    def _get_data(self, index, debug=False):
        # LABEL_INDICES
        name = self.img_names[index]['img_path']
        img_itk = read(self.img_names[index]['img_path'])
        img_ori = itk_to_np(img_itk)
        # spacing = img_itk.GetSpacing()
        scan_orientation = np.argmin(img_ori.shape)
        label_ori = itk_to_np(read(self.img_names[index]['label_path']))

        if min(img_ori.shape) * 2 < max(img_ori.shape):
            orientation = scan_orientation
        else:
            orientation = np.random.randint(3)
        slice_idx = np.random.randint(2, img_ori.shape[orientation]-2)
        if orientation == 0:
            s = img_ori[slice_idx-1:slice_idx+2, :,:]
            lb = label_ori[slice_idx-1:slice_idx+2, :,:]
            # spacing = (spacing[1], spacing[2])
        if orientation == 1:
            s = img_ori[:,slice_idx-1:slice_idx+2,:]
            s = rearrange(s, "h c w -> c h w")
            lb = label_ori[:,slice_idx-1:slice_idx+2,:]
            lb = rearrange(lb, "h c w -> c h w")
            # spacing = (spacing[0], spacing[2])
        if orientation == 2:
            s = img_ori[:,:,slice_idx-1:slice_idx+2]
            s = rearrange(s, "h w c -> c h w")
            lb = label_ori[:,:,slice_idx-1:slice_idx+2]
            lb = rearrange(lb, "h w c -> c h w")
            # spacing = (spacing[0], spacing[1])
        assert s.shape[0] == 3

        if np.float32(lb[1,:,:]>0).sum() <= 200:
            return self._get_data((index+1)%len(self))
        # Choose one label
        label_num = int(lb.max())
        is_good_mask = []
        for label_idx in range(1,label_num+1):
            one_lb = np.float32(lb==label_idx)
            is_good_mask.append(one_lb.sum()>=50)
        label_idx = np.random.choice(range(1,label_num+1), p=np.array(is_good_mask)/np.sum(is_good_mask))
        lb = np.float32(lb==label_idx)
        return s, lb, name, label_idx
    
    # @tfunctime
    def _get_cached_data(self, index):
        name = self.img_names[index]
        img = cv2.imread(name)

        compressed = np.load(name.replace("image/image_", "label/label_").replace(".jpg", ".npz"))
        csr = csr_matrix((compressed['data'], compressed['indices'], compressed['indptr']), shape=compressed['shape'])
        label_ori = csr.toarray()
        label_ori = rearrange(label_ori, "n (c h w) -> n c h w", c=3, h=1024, w=1024)

        label_idx = np.random.randint(0, label_ori.shape[0])
        label_ori = label_ori[label_idx]        
        return rearrange(img, "h w c -> c h w"), label_ori, name, -1

    def to_RGB(self, img):
        # transform images to RGB style
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(int)
        return img

    # @tfunctime
    def __getitem__(self, index, debug=False):
        img_ori, label_ori, name, label_idx = self._get_data(index)
        img_ori = np.clip(img_ori, -200,400)
        img_rgb = self.to_RGB(img_ori)

        assert len(img_rgb.shape) == 3, f"{__file__} Got{img_rgb.shape}"
        bundle_ori = {"img":torch.Tensor(img_rgb).unsqueeze(0), "label":torch.Tensor(label_ori).unsqueeze(0)}
        # import ipdb; ipdb.set_trace()
        if self.is_train:
            # bundle = self.transform(bundle_ori)[0] # use with transforms.RandCropByPosNegLabeld
            bundle = self.transform(bundle_ori)
        else:
            bundle = self.test_transform(bundle_ori)
        
        if not self.use_cache:
            bundle['label'] = (bundle['label']>0.5).float()
        vector = np.ones(3)
        if debug:
            ret_dict = {
            "name": name,
            "img": bundle['img'],
            "label": bundle['label'],
            "img_ori":img_ori,
            "label_ori":label_ori,
            "label_idx": label_idx,
            "indicators": vector,
            # "label_name": 
            }  
            return ret_dict

        ret_dict = {
            "name": name,
            "img": bundle['img'][0].float(),
            "label": bundle['label'][0].float(),
            "indicators": vector,
        }        
        if bundle['label'][0][1,:,:].sum() <= 0:
            return self.__getitem__(index+1 % len(self))
        return ret_dict

        
if __name__ == "__main__":
    from tutils.new.manager import ConfigManager
    config  = ConfigManager()
    config.add_config("configs/vit_b_103.yaml")
    dataset = Dataset3D(config=config['dataset']) # , use_cache=True
    data = dataset.__getitem__(0)

    import ipdb; ipdb.set_trace()
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8)
    for batch in loader:
        print(batch['img'].shape, batch['label'].shape)
        # print(data['label'].max())
        # import ipdb; ipdb.set_trace()
                
