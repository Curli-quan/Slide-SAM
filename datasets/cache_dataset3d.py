"""
    Slow Loading directly

    So we pre-precess data 
"""

import numpy as np
import os
from einops import rearrange, reduce, repeat
from tutils.nn.data import read, itk_to_np, np_to_itk, write
from tutils import tfilename
from .dataset3d import DATASET_CONFIG, Dataset3D as basic_3d_dataset
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

# "class": ["spleen", "right kidney", "left kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "postcava", "portal vein and splenic vein", "pancrease", "right adrenal gland", "left adrenal gland"],
# "class": ["liver", "right kidney", "left kidney", "spleen"],
TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '02': [1,0,3,4,5,6,7,0,0,0,11,0,0,14],
    '03': [6],
    '04': [6,27],       # post process
    '05': [2,26,32],       # post process
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,20,21,23,24],
    '08': [6, 2, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,2],  
    '13': [6,2,1,11,8,9,7,4,5,12,13,25], 
    '14': [11,11,28,28,28],     # Felix data, post process
    '10_03': [6, 27],   # post process
    '10_06': [30],
    '10_07': [11, 28],  # post process
    '10_08': [15, 29],  # post process
    '10_09': [1],
    '10_10': [31],
    '58': [6,2,3,1],
    '59': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '60': np.arange(200).tolist(), # for debug
}

class Dataset3D(basic_3d_dataset):
    def __init__(self, config=..., use_cache=True, *args, **kwargs) -> None:
        super().__init__(config, use_cache=use_cache, *args, **kwargs)
        self.basic_dir = config['data_root_path']
        self.cache_dir = config['cache_data_path']

    def prepare_transforms(self): 
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024,1024)),
        ])
        self.test_transform = transforms.Compose([
            monai_transforms.Resized(keys=['img', 'label'], spatial_size=(3,1024,1024)),            
        ])

    # @tfunctime
    # def prepare_datalist(self):
    def prepare_cached_datalist(self):
        raise DeprecationWarning("[Warning] Please use cache_dataset3d new version instead!")
        config = self.config
        data_paths = []
        for dirpath in glob.glob(config['cache_data_path'] + "/*"):
            data_paths += glob.glob(dirpath + "/image/*.jpg")
            print("Load ", dirpath)
        print('train len {}'.format(len(data_paths)))
        print('Examples: ', data_paths[:2])
        return data_paths  

    def caching_data(self):
        assert self.use_cache == False
        for index in range(len(self)):
            self.cache_one_sample(index)

    def cache_one_sample(self, index, debug=False):
        # LABEL_INDICES
        name = self.img_names[index]['img_path']
        img_itk = read(self.img_names[index]['img_path'])
        img_ori = itk_to_np(img_itk)

        img_ori = np.clip(img_ori, -200,400)

        # spacing = img_itk.GetSpacing()
        scan_orientation = np.argmin(img_ori.shape)
        label_ori = itk_to_np(read(self.img_names[index]['label_path']))
        
        dataset_name = self.img_names[index]['img_path'].replace(self.basic_dir,"").split("/")[0]
        assert dataset_name[0] in ['0','1','2','3','4','5','6','7','8','9'], f"Got {dataset_name}"
        all_labels = TEMPLATE[dataset_name[:2]]

        num = 0

        # if min(img_ori.shape) * 1.2 < max(img_ori.shape):
        #     orientation_all = [scan_orientation]
        # else:
        #     orientation_all = [0,1,2]
        orientation_all = [scan_orientation]

        for orientation in orientation_all:
            for slice_idx in range(2, img_ori.shape[orientation]-2):
                # slice_idx = np.random.randint(2, img_ori.shape[orientation]-2)
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

                # if np.float32(lb[1,:,:]>0).sum() <= 200:
                #     # return self._get_data((index+1)%len(self))
                #     continue
                # Choose one label
                label_num = int(lb.max())

                masks_data = []
                meta = {"img_name": name, "slice": slice_idx, "orientation": orientation, "label_idx": [], "labels": [], "id": f"{num:08d}" }
                for label_idx in range(1,label_num+1):
                    one_lb = np.float32(lb==label_idx)
                    if one_lb[1,:,:].sum() <= (one_lb.shape[-1] * one_lb.shape[-2] * 0.0014):
                        continue
                    # if one_lb[0,:,:].sum()<=50 or one_lb[2,:,:].sum()<=50:
                        
                    masks_data.append(one_lb)
                    meta['label_idx'].append(label_idx)
                    meta['labels'].append(all_labels[label_idx-1])
                
                if len(masks_data) <= 0:
                    continue
                
                img_rgb = s
                img_rgb = F.interpolate(torch.Tensor(img_rgb).unsqueeze(0), size=(1024,1024)).squeeze().numpy()
                img_rgb = self.to_RGB(img_rgb)
                save_image_name = tfilename(self.cache_dir, dataset_name, f"image/image_{index:04d}_{num:08d}.jpg")
                self.save_img_rgb(rearrange(img_rgb, "c h w -> h w c"), save_image_name)

                # Save cache data
                save_label_name = tfilename(self.cache_dir, dataset_name, f"label/label_{index:04d}_{num:08d}.npz")
                self.save_slice_mask(masks_data, save_label_name)
                print("Save ", save_image_name)

                self.save_meta(meta, tfilename(self.cache_dir, dataset_name, f"meta/meta_{index:04d}_{num:08d}.npy"))

                num += 1

    def save_meta(self, meta, path):
        assert path.endswith(".npy")
        np.save(path, meta)

    def save_slice_mask(self, masks_data, prefix):
        masks_data = F.interpolate(torch.Tensor(masks_data), size=(1024,1024)).numpy()
        assert masks_data.shape[1:] == (3,1024,1024), f"{__file__} Got{masks_data.shape}"
        for i in range(masks_data.shape[0]):
            labeli = masks_data[i].astype(np.uint8) * 255
            assert labeli.sum() > 0
            path = tfilename(prefix+f"_{i:04d}.jpg")
            cv2.imwrite(path, rearrange(labeli, "c h w -> h w c"))     
            print("save to ", path)  
    
    def _old_save_slice_mask(self, masks_data, path):
        raise DeprecationWarning()
        exit(0)
        assert path.endswith(".npz")
        # masks_data = np.array([m['segmentation'] for m in masks]).astype(int)
        masks_data = F.interpolate(torch.Tensor(masks_data), size=(1024,1024)).numpy()
        # masks_data = np.int8(masks_data>0)
        assert masks_data.shape[1:] == (3,1024,1024), f"{__file__} Got{masks_data.shape}"
        masks_data = rearrange(masks_data, "n c h w -> n (c h w)")
        csr = csr_matrix(masks_data)
        np.savez_compressed(path, data=csr.data, indices=csr.indices, indptr=csr.indptr, shape=csr.shape)

    def save_img_rgb(self, img, path):
        assert path.endswith(".jpg")
        assert img.shape == (1024,1024,3)
        cv2.imwrite(path, img.astype(np.uint8))

    def _get_cached_data(self, index):
        name = self.img_names[index]
        # print(name)
        img = cv2.imread(name)
        compressed = np.load(name.replace("image/image_", "label/label_").replace(".jpg", ".npz"))
        csr = csr_matrix((compressed['data'], compressed['indices'], compressed['indptr']), shape=compressed['shape'])
        label_ori = csr.toarray()
        label_ori = rearrange(label_ori, "n (c h w) -> n c h w", c=3, h=1024, w=1024)
        meta = np.load(name.replace("image/image_", "meta/meta_").replace(".jpg", ".npy"), allow_pickle=True).tolist()
        # print(meta)
        pp = reduce(label_ori[:,1,:,:], "n h w -> n", reduction="sum") > 500
        if pp.sum() == 0:
            return self._get_cached_data((index+1)%len(self))

        label_idx = np.random.choice(a=np.arange(len(pp)), p=pp/pp.sum())
        # label_idx = np.random.randint(0, label_ori.shape[0])
        label_ori = label_ori[label_idx]        
        is_edge = meta.get('is_edge', 0)
        return rearrange(img, "h w c -> c h w"), label_ori, name, meta['labels'][label_idx], meta['label_idx'][label_idx]

    # @tfunctime
    def __getitem__(self, index, debug=False):
        # print("Dataset warning", index, len(self))
        index = index % len(self)
        img_rgb, label_ori, name, label_idx, local_idx = self._get_cached_data(index)

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
        } 
        return ret_dict

    def _convert_one_mask_from_npz_to_jpg(self, path1=None):
        # path1 = "/home1/quanquan/datasets/cached_dataset2/01_BCV-Abdomen/label/label_0129_00000043.npz" # 32K
        prefix = path1.replace(".npz", "").replace("/label/", "/label_jpg/")
        compressed = np.load(path1)
        csr = csr_matrix((compressed['data'], compressed['indices'], compressed['indptr']), shape=compressed['shape'])
        label_ori = csr.toarray()
        label_ori = rearrange(label_ori, "n (c h w) -> n c h w", c=3, h=1024, w=1024)
        # print(label_ori.shape)
        for i in range(label_ori.shape[0]):
            labeli = label_ori[i]
            path = tfilename(prefix+f"_{i:04d}.jpg")
            cv2.imwrite(path, rearrange(labeli, "c h w -> h w c").astype(np.uint8))     
            print("save to ", path)  

    def convert_masks_types(self):
        assert self.use_cache == True
        for index in range(len(self)):
            name = self.img_names[index]
            label_path = name.replace("image/image_", "label/label_").replace(".jpg", ".npz")
            self._convert_one_mask_from_npz_to_jpg(label_path)

if __name__ == "__main__":
    # def go_cache():
    from tutils.new.manager import ConfigManager
    config  = ConfigManager()
    config.add_config("configs/vit_b.yaml")
    dataset = Dataset3D(config=config['dataset'], use_cache=True)
    dataset.caching_data()
    # dataset.convert_masks_types()
