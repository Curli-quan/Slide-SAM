import torch
import numpy as np
from tutils import tfilename
from tutils.nn.data import read, itk_to_np, np_to_itk, write
from torchvision.utils import save_image
import SimpleITK as sitk
from tutils.nn.data.tsitk.preprocess import resampleImage


class Data3dSolver:
    def __init__(self) -> None:
        pass

    def simple_write(self, data_np, path="tmp.nii.gz", spacing=None):
        assert len(data_np.shape) == 3, f"Got {data_np.shape}"
        data_np = data_np.astype(np.int16)
        data_itk = np_to_itk(data_np)
        if spacing is not None:
            data_itk.SetSpacing(spacing)
        write(data_itk, path=tfilename(path))
        print("Save to ", path)

    def write_slices(self, data, path="tmp_masks.jpg"):
        if isinstance(data, torch.Tensor):
            pass
        if isinstance(data, np.ndarray):
            data = torch.Tensor(data)
        assert len(data.shape) == 4, f"Shape should be (b c h w) c=1/3, Got {data.shape}"
        assert data.shape[1] == 1 or data.shape[1] == 3, f"Shape should be (b c h w) c=1/3, Got {data.shape}"
        assert path.endswith(".jpg") or path.endswith(".png")
        save_image(torch.Tensor(data).unsqueeze(1), tfilename(path))
        print("Save to ", path)

    def write_multilabel_nii(self, data, path, meta=None):
        if isinstance(data, dict):
            data_all = [v for k,v in data.items()]
            data = np.stack(data_all, axis=0)            
        assert len(data.shape) == 4, f"Shape should be (b c h w) , Got {data.shape}"
        # Merge labels to one
        merged = np.zeros_like(data[0])
        for i, datai in enumerate(data):
            merged = np.where(datai > 0, datai * (i+1), merged)

        merged = merged.astype(np.int16)
        data_itk = np_to_itk(merged)
        if meta is not None:
            data_itk = formalize(data_itk, meta)
        write(data_itk, path=tfilename(path))
        print("Save to ", path)
        
    def fwrite(self, data, path, meta):
        data = data.astype(np.int16)
        data_itk = np_to_itk(data)
        data_itk = formalize(data_itk, meta)
        write(data_itk, path=tfilename(path))

    def read(self, path, spacing_norm=True):
        data_itk = read(path)
        if spacing_norm:
            ori_size = data_itk.GetSize()
            ori_spacing = data_itk.GetSpacing()
            data_itk = self.normalize_spacing(data_itk)
            new_size = data_itk.GetSize()
            new_spacing = data_itk.GetSpacing()
            print("Change size from ", ori_size, new_size)
            print("Change spacing from ", ori_spacing, new_spacing)
        data_np = itk_to_np(data_itk)
        print("[data_utils.DEBUG]", data_np.shape)
        return data_np, data_itk.GetSpacing()

    def normalize_spacing(self, data_itk):
        spacing = data_itk.GetSpacing()
        new_spacing = (min(spacing),min(spacing),min(spacing))
        data_itk = resampleImage(data_itk, NewSpacing=new_spacing)
        return data_itk


def formalize(img:sitk.SimpleITK.Image, meta:sitk.SimpleITK.Image):
    # Size = meta.GetSize()
    Spacing = meta.GetSpacing()
    Origin = meta.GetOrigin()
    Direction = meta.GetDirection()
    
    img.SetSpacing(Spacing)
    img.SetOrigin(Origin)
    img.SetDirection(Direction)
    return img

    
def write(img:sitk.SimpleITK.Image, path:str, mode:str="nifti"):
    """
    Path: (example) os.path.join(jpg_dir, f"trans_{random_name}.nii.gz")
    """
    mode = mode.lower()
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(img)