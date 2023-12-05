# from torchvision import transforms
from monai import transforms
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
import glob
import os
from einops import rearrange, repeat
from tutils.nn.data.tsitk import read
from tqdm import tqdm
# from monai.transforms import SpatialPadd, CenterSpatialCropd, Resized, NormalizeIntensityd
# from monai.transforms import RandAdjustContrastd, RandShiftIntensityd, Rotated, RandAffined
# from datasets.common_2d_aug import RandomRotation, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from tutils import tfilename, tdir
import random
from tutils.nn.data import itk_to_np
from scipy.sparse import csr_matrix
import cv2


DEFAULT_PATH="/quanquan/datasets/08_AbdomenCT-1K/"


class Dataset2D(dataset):
    def __init__(self, dirpath=None, is_train=True, getting_multi_mask=False) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.is_train = is_train
        self.getting_multi_mask = getting_multi_mask
        self.img_names = self.prepare_datalist()
        self.prepare_transforms()
        self.weights_dict = {"gt":2, "sam_auto_seg":2, "prompt_point_from_superpixel":1, "prompt_box_from_superpixel":1, "superpixel":0}

    def prepare_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resized(keys=['img', 'label'], spatial_size=(3,1024,1024)),
            # transforms.RandSpatialCropd(keys=["img"], roi_size=(448,448,1)),
            transforms.RandAffined(keys=['img', 'label'], prob=0.5, shear_range=(0.2,0.2)),
            transforms.RandCropByPosNegLabeld(keys=['img', 'label'], spatial_size=(3,960,960), label_key='label', neg=0),
            # transforms.RandSmoothFieldAdjustContrastd(keys=['img', 'label'], )
            transforms.Resized(keys=['img', 'label'], spatial_size=(3,1024,1024)),
            transforms.RandAdjustContrastd(keys=['img'], ),
            transforms.RandShiftIntensityd(keys=['img'], prob=0.8, offsets=(-5, 5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resized(keys=['img'], spatial_size=(3,1024,1024)),            
        ])

    def __len__(self):
        return len(self.img_names)

    def to_RGB(self, img):
        # transform images to RGB style
        img = ((img - img.min()) / img.max() * 255).astype(int)
        return img
    
    def prepare_datalist(self):
        dirpath_img = os.path.join(self.dirpath, 'preprocessed', 'cache_2d_various_pseudo_masks')
        names = glob.glob(os.path.join(dirpath_img, "*_mask.npz"))
        names = [os.path.split(name)[-1].replace("_mask.npz", "") for name in names]
        names.sort()
        # names = names[:15000]
        print(f"[Dataset2d] Load {len(names)} paths.")
        assert len(names) > 0, f"{__file__} Gotdirpath: {self.dirpath}"
        return names
    
    
    def _get_data(self, index, debug=False, iternum=0):
        img_names = self.img_names
        img_info = os.path.split(img_names[index])[-1].split('_s')
        filename, slice_idx = img_info[0], int(img_info[-1][:4])
        mask_loc = np.random.randint(0,3)
        if mask_loc == 0:
            slices_indices = [slice_idx, slice_idx+1, slice_idx+2]
        elif mask_loc == 1:
            slices_indices = [slice_idx-1, slice_idx, slice_idx+1]
        elif mask_loc == 2:
            slices_indices = [slice_idx-2, slice_idx-1, slice_idx]
        
        # Load .npy data
        filenames = [os.path.join(self.dirpath, "preprocessed", "cache_jpg", f"{filename}_s{i:04}_img.jpg") for i in slices_indices]
        for name in filenames:
            if not os.path.exists(name):
                return self._get_data(index+1 % len(self))
            
        imgs = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in filenames]
        img_rgb = np.stack(imgs, axis=0)

        # Load RGB data

        compressed = np.load(os.path.join(self.dirpath, "preprocessed", "cache_2d_various_pseudo_masks", img_names[index]+"_mask.npz"))
        csr = csr_matrix((compressed['data'], compressed['indices'], compressed['indptr']), shape=compressed['shape'])
        label_ori = csr.toarray()
        label_ori = rearrange(label_ori, "c (h w) -> c h w", h=1024, w=1024)
        metadata = np.load(os.path.join(self.dirpath, "preprocessed", "cache_2d_various_pseudo_masks", img_names[index]+"_metadata.npy"), allow_pickle=True)

        label_prob = np.array([self.weights_dict[item['source']] for item in metadata]).astype(float)
        label_prob = label_prob / label_prob.sum()

        label_idx = np.random.choice(a=np.arange(len(metadata)), p=label_prob)
        label_ori = label_ori[label_idx]        
        metadata = metadata[label_idx]
        assert metadata['source'] != 'superpixel'

        assert len(img_rgb.shape) == 3, f"{__file__} Got{img_rgb.shape}"
        bundle_ori = {"img":torch.Tensor(rearrange(img_rgb, "c h w -> 1 c h w")), "label":torch.Tensor(repeat(label_ori, "h w -> 1 3 h w"))}
        # import ipdb; ipdb.set_trace()
        if self.is_train:
            bundle = self.transform(bundle_ori)[0]
        else:
            bundle = self.test_transform(bundle_ori)
            
        bundle['label'] = (bundle['label']>0.5).float()
        if bundle['label'][0].sum() < 100:
            return self._get_data((index+1)%len(self), iternum=iternum+1)

        vector = np.zeros(3)
        vector[mask_loc] = 1

        if debug:
            ret_dict = {
            "name": img_names[index],
            "img": bundle['img'][0],
            "label": bundle['label'][0],
            "img_ori":img_rgb,
            "label_ori":label_ori,
            "weight": self.weights_dict[metadata['source']],
            "iternum": iternum,
            "mask_loc": mask_loc,
            "indicators": vector,
            }  
            return ret_dict

        ret_dict = {
            "name": img_names[index],
            "img": bundle['img'][0],
            "label": bundle['label'][0],
            "mask_loc": mask_loc,
            "indicators": vector,
        }        
        return ret_dict

    
    def __getitem__(self, index):
        return self._get_data(index)


class Testset2d(Dataset2D):
    def __init__(self, dirpath=None, is_train=False, getting_multi_mask=False) -> None:
        super().__init__(dirpath, is_train, getting_multi_mask)
        self.test_names = self.prepare_datalist()

    def prepare_datalist(self):
        dirpath_img = os.path.join(self.dirpath, 'cache_2d_various_pseudo_masks')
        names = glob.glob(os.path.join(dirpath_img, "*_mask.npz"))
        names = [os.path.split(name)[-1].replace("_mask.npz", "") for name in names]
        names.sort()
        names = names[15000:]
        print(f"[Dataset2d] Load {len(names)} paths.")
        assert len(names) > 0, f"{__file__} Gotdirpath: {self.dirpath}"
        return names

        
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Dataset2D(dirpath=DEFAULT_PATH)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    iternums = 0
    for i, data in enumerate(loader):
        # iternums += data['iternum'].item()
        print(i, iternums / (i+1), data['img'].shape, data['label'].shape)
        assert data['label'].sum() >= 100, f"{__file__} Got{data['label'].sum()}"
        assert torch.Tensor(data['label']==1).sum() >= 100, f"{__file__} Got {torch.Tensor(data['label']==1).sum().sum()}"

    import ipdb; ipdb.set_trace()
                
