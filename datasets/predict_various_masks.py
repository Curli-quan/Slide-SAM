# from utils.predict_automasks import predict
from segment_anything import sam_model_registry, SamPredictor
from core.automask import SamAutomaticMaskGenerator
from core.trainer import SamLearner
# from datasets.dataset2d import Dataset2D
from einops import rearrange, repeat
import torch
import numpy as np

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import os
from tutils import timer, tdir
from scipy.sparse import csr_matrix
# from datasets.dataset2d import Dataset2D
import cv2
import torch.nn.functional as F
import time
from tutils import tfilename

import matplotlib.pylab as plt
from .utils import load_compressed_data, show_anns, img_to_show
import cv2
        
def tfunctime(func):
    def run(*argv, **kargs):
        t1 = time.time()
        ret = func(*argv, **kargs)
        t2 = time.time()
        print(f"[Function <{func.__name__}>] Running time:{(t2-t1):.6f}s")
        return ret
    return run

def get_predictors():
    sam_checkpoint = "/quanquan/code/segment-anything/segment_anything/sam_vit_h_4b8939.pth" # for A100
    # sam_checkpoint = "/home1/quanquan/code/projects/medical-guangdong/segment-anything/sam_vit_h_4b8939.pth" # for server 103
    device = "cuda"
    model_type = "default"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamLearner(sam_model=sam)
    return mask_generator, predictor

def center_clip(img, eta=3):
    count_values = img[torch.where(img>-199)]
    mean = count_values.mean()
    std = count_values.std()
    img3 = torch.clip(img, mean-eta*std, mean+eta*std)
    
    return img3


def find_not_exist_masks(masks_repo, masks_new, iou_threshold=0.2):
    if len(masks_repo) == 0:
        return masks_new    
    def compute_iou(m1, m2):
        m1 = m1['segmentation']
        m2 = m2['segmentation']
        intersection = m1*m2
        union = np.float32((m1 + m2) > 0)
        return intersection.sum() / union.sum()
    to_append = []
    for mask_new in masks_new:
        assert isinstance(mask_new, dict), f"{__file__} Got{type(mask_new)}"
        intersec_count = 0
        for mask_in_repo in masks_repo:
            assert isinstance(mask_in_repo, dict), f"{__file__} Got{type(mask_in_repo)}"
            iou = compute_iou(mask_in_repo, mask_new)
            .3
            if iou > iou_threshold:
                intersec_count += 1
        if intersec_count == 0:
            to_append.append(mask_new)
    check_keys(to_append)
    return to_append


def merge_masks(masks_repo, masks_new, iou_threshold=0.2):
    to_append = find_not_exist_masks(masks_repo, masks_new, iou_threshold)
    # print(f"DEBUG: {len(masks_new) - len(to_append)} masks are deleted, remaining {len(to_append)}. The total achieves {len(masks_repo) + len(to_append)}")
    return masks_repo + to_append


def dilate_erode(mask):
    kernel = np.ones((4, 4), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(float), cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def get_superpixel(image, hu_mask, hu_threshold=-50):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(hu_mask, torch.Tensor):
        hu_mask = hu_mask.detach().cpu().numpy()
    segments = slic(image, n_segments=100, compactness=9)
    mask_collect = []
    image2 = image[:,:,0]
    for i in range(1, segments.max()+1):
        mask = torch.Tensor(segments==i).detach().cpu().numpy()
        # assert img
        if image2[segments==i].mean() <= hu_threshold:
            continue
        mask = mask * hu_mask
        mask = dilate_erode(mask)
        mask_data = {
            'segmentation':mask,
            'area':mask.sum(),
            'source': "superpixel",
            'mean_value':(mask * image[:,:,0]).mean(),
        }
        mask_collect.append(mask_data)
    check_keys(mask_collect)
    return mask_collect

# def resize_masks(masks):
#     for m in masks:
#         m['segmentation'] = F.interpolate(torch.Tensor(m['segmentation'])[None,None,:,:], size=(512,512)).squeeze().numpy()
#     return masks

# @tfunctime
def get_masks_via_color_changing(img, label, mask_generator, predictor=None):
    img = repeat(img, " h w -> 1 3 h w")
    img = torch.Tensor(img)
    img  = F.interpolate(img, size=(1024,1024))

    label = torch.Tensor(label)
    if label is not None:
        label_masks = []
        for i in range(1,label.max().int()+1):
            labeli = torch.Tensor(label==i).float()
            labeli = F.interpolate(labeli[None, None, :,:], size=(1024,1024)).squeeze().numpy()
            area = labeli.sum() 
            if area <=10:
                continue
            mask = {
                "segmentation":labeli,
                "area":area,
                "source": "gt",
            }
            label_masks.append(mask)
    else:
        label_masks = []

    mask_generator.reset_image()
    predictor.reset_image()

    masks = mask_generator.generate(img.cuda())
    masks = filter_large_masks(masks)
    for mask in masks:
        mask['source'] = "sam_auto_seg"
    label_masks = merge_masks(label_masks, masks)
    del masks

    check_keys(label_masks)
    # import ipdb; ipdb.set_trace()
    # return img, label_masks

    img2 = center_clip(img, 2.5)
    masks = mask_generator.generate(img.cuda())
    masks = filter_large_masks(masks)
    label_masks = merge_masks(label_masks, masks)
    del masks
    del img2

    # img2 = center_clip(img, 2)
    # masks = mask_generator.generate(img2.cuda())
    # masks = filter_large_masks(masks)
    # for mask in masks:
    #     mask['source'] = "sam_auto_seg"
    # label_masks = merge_masks(label_masks, masks)
    # del masks
    # del img2
    
    img2 = center_clip(img, 1)
    masks = mask_generator.generate(img2.cuda())
    masks = filter_large_masks(masks)
    for mask in masks:
        mask['source'] = "sam_auto_seg"
    label_masks = merge_masks(label_masks, masks)
    del masks
    del img2
    
    img2 = center_clip(img, 0.5)
    masks = mask_generator.generate(img2.cuda())
    masks = filter_large_masks(masks)
    for mask in masks:
        mask['source'] = "sam_auto_seg"
    label_masks = merge_masks(label_masks, masks)
    del masks
    del img2

    check_keys(label_masks)
    # import ipdb; ipdb.set_trace()
    return img, label_masks

def check_keys(masks):
    for m in masks:
        assert m['segmentation'] is not None
        assert m['area'] is not None
        assert m['source'] is not None

def filter_large_masks(masks):
    filtered_masks = []
    for mask in masks:
        if mask['area'] > 0.25 *1024 * 1024:
            continue
        filtered_masks.append(mask)
    del masks
    return filtered_masks


def mix_masks(masks):
    if len(masks) == 0:
        return None
    mixed_mask = None
    for item in masks:
        m = item['segmentation']
        mixed_mask = np.zeros_like(m) if mixed_mask is None else mixed_mask
        mixed_mask += m
    mixed_mask = np.float32(mixed_mask>0)
    return mixed_mask


def select_random_point_from_mask(gt_mask):
    size = gt_mask.shape
    assert len(size) == 2
    xy = np.arange(0, size[0] * size[1])
    gt_mask = np.float32(gt_mask>0)
    prob = rearrange(gt_mask, "h w -> (h w)")
    prob = prob / prob.sum()
    loc = np.random.choice(a=xy, size=1, replace=True, p=prob)[0]
    x, y = loc % size[1], loc // size[1]
    return x, y


def select_center_point_from_mask(gt_mask):
    # get indices of all the foreground pixels
    indices = np.argwhere(gt_mask > 0)
    # calculate the center point by taking the mean of the foreground pixel indices
    center = np.mean(indices, axis=0).astype(int)
    y, x = center
    return x, y

@tfunctime
def get_masks_via_points_from_superpixels(img, label_masks, superpixels, predictor, hu_mask=None):
    if isinstance(hu_mask, torch.Tensor):
        hu_mask = hu_mask.detach().cpu().numpy()
    total_mask = mix_masks(label_masks)
    # superpixels = get_superpixel(rearrange(img, "1 c h w -> h w c"))
    points = []
    ex_masks = []
    for seg in superpixels:
        mask_collect = []
        for i in range(5):
            mm = np.nan_to_num(seg['segmentation'], 0)
            if mm.sum() <= 0:
                continue
            x,y = select_center_point_from_mask(mm)   # select_random_point_from_mask
            # print(x,y) 
            if total_mask[y,x] == 1:
                continue
            # else:
            #     print(total_mask[y,x])
            point = torch.Tensor([x,y])
            points.append(point)
            mask = predictor.generate(img.cuda(), point.cuda().unsqueeze(0).unsqueeze(0))[0,0].detach().cpu().numpy()
            mask = mask * hu_mask
            mask = dilate_erode(mask)
            mask = {
                "segmentation": mask,
                "area": mask.sum(),
                "source": "prompt_point_from_superpixel",
            }
            mask_collect = merge_masks(mask_collect, [mask])
        ex_masks += mask_collect
    check_keys(ex_masks)
    return ex_masks

def mask_to_bbox(mask):
    """ copied from data_engine """
    # Find the indices of all non-zero elements in the mask
    coords = np.nonzero(mask)

    # Compute the minimum and maximum values of the row and column indices
    x_min = np.min(coords[1])
    y_min = np.min(coords[0])
    x_max = np.max(coords[1])
    y_max = np.max(coords[0])

    # Return the coordinates of the bounding box
    return (x_min, y_min, x_max, y_max)

@tfunctime
def get_masks_via_boxes_from_superpixels(img, label_masks, superpixels, predictor, hu_mask=None):
    if isinstance(hu_mask, torch.Tensor):
        hu_mask = hu_mask.detach().cpu().numpy()
    ex_masks = []
    for seg in superpixels:
        if seg['segmentation'].sum() < 100:
            continue
        x_min, y_min, x_max, y_max = mask_to_bbox(seg['segmentation'])   
        box = torch.Tensor([x_min, y_min, x_max, y_max])
        mask = predictor.generate_by_box(img.cuda(), box.cuda().unsqueeze(0).unsqueeze(0))[0,0].detach().cpu().numpy()
        mask = mask * hu_mask
        mask = dilate_erode(mask)
        mask = {
            "segmentation": mask,
            "area": mask.sum(),
            "source": "prompt_box_from_superpixel",
        }
        ex_masks = merge_masks(ex_masks, [mask])
    check_keys(ex_masks)
    return ex_masks 


class VariousPseudoMasksGenerator:
    def __init__(self, dataset, label_path:str=None) -> None:
        self.dataset = dataset
        # assert dirpath.split("/")[-1] == "cache_2d", f"{__file__} Got{dirpath.split('/')[-1]} ; dirpath:{dirpath}"
        self.label_path = label_path if label_path is not None else dataset.dirpath.replace("cache_2d", "cache_2d_various_pseudo_masks")

    def example(self, mask_generator=None, predictor=None):
        return self.generate(mask_generator=mask_generator, predictor=predictor, is_example=True)

    def generate(self, mask_generator=None, predictor=None, is_example=False):
        tt = timer()
        if mask_generator is None:
            mask_generator, predictor = get_predictors()
            self.mask_generator, self.predictor = mask_generator, predictor
        for img_idx in range(len(self.dataset)):
            tt()
            data = self.dataset._get_image(img_idx)       
            masks_all_layers = []
            words = data['name'].split("/")
            dataset_name = words[0]
            filename = words[-1].replace(".nii.gz","")
            volume_rgb = np.clip(data['img'], -200, 400)
            volume_rgb = (volume_rgb - volume_rgb.min()) / volume_rgb.max()
            for slice_idx in range(data['img'].shape[0]):
                path = tfilename(self.label_path, f'{dataset_name}/{filename}_s{slice_idx}_mask.npz')
                # if os.path.exists(path):
                #     continue
                masks = self.get_various_masks(data['img'][slice_idx], data['label'][slice_idx], mask_generator, predictor)
                self.save_slice_mask(masks, path, slice_idx)
                img_path = tfilename(self.label_path, f'image_{dataset_name}/{filename}_s{slice_idx}.jpg')
                self.save_img_rgb(volume_rgb[slice_idx], img_path)
                # display
                # plt.figure(figsize=(6,6))
                # plt.imshow(img_to_show(data['img'][slice_idx]), cmap='gray')
                # show_anns(masks)
                # plt.axis('off')
                # plt.show() 
                print(f"Save to {img_path} and {path}")
                
            print(f"Processing {img_idx}, {len(masks_all_layers)} saved, time used:{tt()}", end='\r')
            if is_example:
                break

    def save_slice_mask(self, masks, path, slice_idx):
        masks_data = np.array([m['segmentation'] for m in masks]).astype(int)
        # if len(masks_data) <= 1:
        #     import ipdb; ipdb.set_trace()
        masks_data = F.interpolate(torch.Tensor(masks_data).unsqueeze(1), size=(512,512)).squeeze().numpy()
        masks_data = np.int8(masks_data>0)
        if len(masks_data.shape) == 2:
            masks_data = masks_data[None,:,:]
        assert masks_data.shape[1:] == (512,512), f"{__file__} Got{masks_data.shape}"
        masks_data = rearrange(masks_data, "n h w -> n (h w)")
        csr = csr_matrix(masks_data)
        np.savez_compressed(path, data=csr.data, indices=csr.indices, indptr=csr.indptr, shape=csr.shape)

    def save_img_rgb(self, img, path):
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(path, img)


    @staticmethod
    @tfunctime
    def get_various_masks(img_ori, label, mask_generator, predictor):
        mask_generator.reset_image()
        predictor.reset_image()
        img, label_masks = get_masks_via_color_changing(img_ori, label, mask_generator, predictor)

        # # return label_masks
        # hu_mask = img_ori>img_ori.min() + 10
        # hu_mask = F.interpolate(torch.Tensor(hu_mask)[None,None,:,:], size=(1024,1024)).squeeze()

        # superpixels = get_superpixel(rearrange(img, "1 c h w -> h w c"), hu_mask=hu_mask)
        # exclu_superpixels = find_not_exist_masks(label_masks, superpixels)
        # # point_prompt_masks = get_masks_via_boxes_from_superpixels(img, label_masks, exclu_superpixels, predictor, hu_mask=hu_mask)
        # box_prompt_masks = get_masks_via_points_from_superpixels(img, label_masks, exclu_superpixels, predictor, hu_mask=hu_mask)
        
        return label_masks # + box_prompt_masks # + point_prompt_masks #


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=6 python -m datasets.predict_various_masks
    from datasets.dataset3d import Dataset3D
    dataset = Dataset3D()
    gen = VariousPseudoMasksGenerator(dataset=dataset,
                                      label_path=tdir("/quanquan/datasets/all_datasets/various_masks_3/"))
    gen.generate()

    