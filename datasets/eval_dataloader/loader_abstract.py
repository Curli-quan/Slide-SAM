"""
    DataLoader only for evaluation
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tutils.nn.data import read, itk_to_np, np_to_itk
from einops import reduce, repeat, rearrange
# from tutils.nn.data.tsitk.preprocess import resampleImage
from trans_utils.data_utils import Data3dSolver
from tutils.nn.data.tsitk.preprocess import resampleImage
import SimpleITK as sitk


# Example
DATASET_CONFIG={
    'split': 'test',
    'data_root_path':'/quanquan/datasets/',
    'dataset_list': ["ours"],
    'data_txt_path':'./datasets/dataset_list/',
    'label_idx': 0,
}

class AbstractLoader(Dataset):
    def __init__(self, config, split="test") -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.img_names = self.prepare_datalist() 
    
    def __len__(self):
        return len(self.img_names)

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


    def _get_data(self, index, debug=False):
        label_idx = self.config['label_idx']
        name = self.img_names[index]['img_path']
        img_itk = read(name)
        spacing = img_itk.GetSpacing()
        img_ori = itk_to_np(img_itk)
        scan_orientation = np.argmin(img_ori.shape)
        label_ori = itk_to_np(read(self.img_names[index]['label_path']))
        label = label_ori == label_idx
        
        # img_ori, new_spacing = Data3dSolver().read(self.img_names[index]['img_path'])
        # label_itk = read(self.img_names[index]['label_path'])
        # ori_spacing = label_itk.GetSpacing()
        # label = itk_to_np(label_itk) == label_idx
        # print("[loader_abstract.DEBUG] size", img_ori.shape, label.shape)
        # label = self._get_resized_label(label, new_size=img_ori.shape)

        if debug:
            Data3dSolver().simple_write(label)
            Data3dSolver().simple_write(img_ori, "tmp_img.nii.gz")

        s = reduce(label, "c h w -> c", reduction="sum")
        coords = np.nonzero(s)
        x_min = np.min(coords[0])
        x_max = np.max(coords[0])
        template_slice_id = s.argmax() - x_min
        
        if img_ori.min() < -10:
            img_ori = np.clip(img_ori, -200, 400)
        else:
            img_ori = np.clip(img_ori, 0, 600)     

        img_ori = img_ori[x_min:x_max+1,:,:]
        label = label[x_min:x_max+1,:,:]
        assert label.shape[0] >= 3

        if template_slice_id <= 1 or template_slice_id >= label.shape[0]-2:
            template_slice_id == label.shape[0] // 2

        dataset_name = name.replace(self.config['data_root_path'], "").split("/")[0]
        template_slice = label[template_slice_id,:,:]
        print("template_slice.area ", template_slice.sum(), template_slice.sum() / (template_slice.shape[0] * template_slice.shape[1]))
        d = {
            "name": name,
            "dataset_name": dataset_name,
            "img": np.array(img_ori).astype(np.float32),
            "label_idx": label_idx,
            "label": np.array(label).astype(np.float32),
            "template_slice_id": template_slice_id,
            "template_slice": np.array(label[template_slice_id,:,:]).astype(np.float32),
            "spacing": np.array(spacing),
            }
        return d

    def __getitem__(self, index):
        return self._get_data(index)


if __name__ == "__main__":
    from tutils.new.manager import ConfigManager
    EX_CONFIG = {       
        'dataset':{
            'prompt': 'box',
            'dataset_list': ['guangdong'], # ["sabs"], chaos, word
            'label_idx': 2,
        }       
    }

    config = ConfigManager()
    config.add_config("configs/vit_sub_rectum.yaml")
    config.add_config(EX_CONFIG)
    dataset = AbstractLoader(config['dataset'], split="test")
    for i in range(len(dataset)):
        dataset._get_data(i, debug=False)

    # label_path = "/home1/quanquan/datasets/01_BCV-Abdomen/Training/label/label0001.nii.gz"
    # from monai.transforms import SpatialResample
    # resample = SpatialResample()
    # label = itk_to_np(read(label_path)) == 1
    # print(label.shape)
    # # resampled = resample(label, spatial_size=(label.shape[0]*7, label.shape[1], label.shape[2]))
    # print(label.shape)

    exit(0)
    data = itk_to_np(read("tmp_img.nii.gz"))
    data = torch.Tensor(data)
    
    maxlen = data.shape[0]
    slices = []
    for i in range(1, maxlen-1):
        slices.append(data[i-1:i+2, :, :])
    input_slices = torch.stack(slices, axis=0)
    input_slices = torch.clip(input_slices, -200, 600)

    input_slices

    from torchvision.utils import save_image
    save_image(input_slices, "tmp.jpg")
