import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from einops import repeat, rearrange

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

class PointPromptGenerator(object):
    def __init__(self, size=None) -> None:
        pass      

    def get_prompt_point(self, gt_mask):
        # assert gt_mask.shape == (1024,1024) or gt_mask.shape == (512,512), f"[data_engine] {__file__} Got{gt_mask.shape}"
        if not (gt_mask.shape == (1024,1024) or gt_mask.shape == (512,512) or gt_mask.shape == (256,256) ): 
            print(f"[Warning] [data_engine] {__file__} Got{gt_mask.shape}")
        assert gt_mask.sum() > 0
        self.size = gt_mask.shape
        self.xy = np.arange(0, self.size[0] * self.size[1])

        gt_mask = np.float32(gt_mask>0)
        prob = rearrange(gt_mask, "h w -> (h w)")
        prob = prob / prob.sum()
        loc = np.random.choice(a=self.xy, size=1, replace=True, p=prob)[0]
        x, y = loc % self.size[1], loc // self.size[1]
        return x, y

    @staticmethod
    def select_random_subsequent_point(pred_mask, gt_mask):
        # union = np.float32((pred_mask + gt_mask)>0)
        # diff = union - intersect
        assert len(pred_mask.shape) == 2
        assert len(gt_mask.shape) == 2
        assert gt_mask.sum() > 0, f"[data_engine] Got {gt_mask.sum()}==0 "
        diff = np.float32(np.abs(pred_mask - gt_mask)>0)
        diff = np.nan_to_num(diff, nan=0)
        # print(diff.shape)
        xy = np.arange(0, diff.shape[0] * diff.shape[1])
        
        if diff.sum() == 0:
            prob = rearrange(gt_mask, "h w -> (h w)")
            prob = prob / prob.sum()
            loc = np.random.choice(a=xy, size=1, replace=True, p=prob)[0]
            x, y = loc % diff.shape[1], loc // diff.shape[1]
            return (x,y), 1
        # Get_prompt_point
        prob = rearrange(diff, "h w -> (h w)")
        prob = prob / prob.sum()
        loc = np.random.choice(a=xy, size=1, replace=True, p=prob)[0]
        x, y = loc % diff.shape[1], loc // diff.shape[1]

        if gt_mask[y, x] == 1 and pred_mask[y, x] == 0:
            classification = 1
        else:
            classification = 0
            # raise ValueError
        return (x, y), classification



class BoxPromptGenerator(object):
    def __init__(self, size) -> None:
        self.size = size

    @staticmethod
    def mask_to_bbox(mask):
        # Find the indices of all non-zero elements in the mask
        coords = np.nonzero(mask)

        # Compute the minimum and maximum values of the row and column indices
        x_min = np.min(coords[1])
        y_min = np.min(coords[0])
        x_max = np.max(coords[1])
        y_max = np.max(coords[0])

        # Return the coordinates of the bounding box
        return (x_min, y_min, x_max, y_max)
        # return (y_min, x_min, y_max, x_max)

    def add_random_noise_to_bbox(self, bbox):
        bbox = list(bbox)
        # Calculate the side lengths of the box in the x and y directions
        x_side_length = bbox[2] - bbox[0]
        y_side_length = bbox[3] - bbox[1]

        # Calculate the standard deviation of the noise
        std_dev = 0.01 * (x_side_length + y_side_length) / 2

        # Generate random noise for each coordinate
        x_noise = np.random.normal(scale=std_dev)
        y_noise = np.random.normal(scale=std_dev)

        # Add the random noise to each coordinate, but make sure it is not larger than 20 pixels
        bbox[0] += min(int(round(x_noise)), 20)
        bbox[1] += min(int(round(y_noise)), 20)
        bbox[2] += min(int(round(x_noise)), 20)
        bbox[3] += min(int(round(y_noise)), 20)

        # Make sure the modified coordinates do not exceed the maximum possible values
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], self.size[0])
        bbox[3] = min(bbox[3], self.size[1])

        # Return the modified bounding box
        return bbox

    def get_prompt_box(self, gt_mask):
        """ return (x_min, y_min, x_max, y_max) """
        assert gt_mask.shape == (1024,1024) or gt_mask.shape == (512,512) or gt_mask.shape == (256,256), f"[data_engine] {__file__} Got{gt_mask.shape}"
        box = self.mask_to_bbox(gt_mask)
        box_w_noise = self.add_random_noise_to_bbox(box)
        return box_w_noise

    def enlarge(self, bbox, margin=0):
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        margin_x = int((x1 - x0)*0.05)
        margin_y = int((y1 - y0)*0.05)
        x0 = max(x0 - margin_x, 0)
        y0 = max(y0 - margin_x, 0)
        x1 = min(x1 - margin_y, self.size[0]-1)
        y1 = min(y1 - margin_y, self.size[1]-1)

        # print("[DEBUG] , enlarge size: ", margin_x, margin_y)
        # print("[DEBUG] from", bbox, "to", (x0,y0,x1,y1))
        return (x0,y0,x1,y1)

class DataEngine(Dataset):
    def __init__(self, dataset=None, img_size=None) -> None:
        # CACHE_DISK_DIR="/home1/quanquan/code/projects/medical-guangdong/cache/data2d_3/"
        super().__init__()    
        self.point_prompt_generator = PointPromptGenerator(img_size)
        self.box_prompt_generator = BoxPromptGenerator(img_size)
        # self._get_dataset(dirpath=dirpath)
        self.dataset = dataset
    
    # def _get_dataset(self, dirpath):
    #     self.dataset = Dataset2D(dirpath=dirpath, is_train=True)    

    def __len__(self):
        return len(self.dataset)
    
    def _get_true_index(self, idx):
        return idx

    def __getitem__(self, idx):
        return self.get_prompt(idx)

    def get_prompt_point(self, gt_mask):
        return self.point_prompt_generator.get_prompt_point(gt_mask)

    def get_prompt_box(self, gt_mask):
        return self.box_prompt_generator.get_prompt_box(gt_mask)

    # def _get_data_from_dataset(self, idx):

    def get_prompt(self, idx):
        idx = self._get_true_index(idx)
        data = self.dataset.__getitem__(idx) 
        
        img = data['img'] # (3,h,w) d=3
        mask = data['label'] # (3,h,w) d=3

        try:
            gt_mask = mask[1,:,:]
        except Exception as e:
            import ipdb; ipdb.set_trace()
        gt_mask = mask[1,:,:]
        gt_mask = gt_mask.numpy() if isinstance(gt_mask, torch.Tensor) else gt_mask
        
        # if np.random.rand() > 0.5:
        prompt_point = self.get_prompt_point(gt_mask)
        # else:
        prompt_box =  self.get_prompt_box(gt_mask)

        data['prompt_point'] = np.array(prompt_point).astype(np.float32)
        data['prompt_box'] = np.array(prompt_box).astype(np.float32)
        data['point_label'] = np.ones((1,)).astype(np.float32)
        return data
        
    def get_subsequent_prompt_point(self, pred_mask, gt_mask):
        # return self.point_prompt_generator.select_random_subsequent_point_torch(pred_mask, gt_mask)
        # return self.point_prompt_generator.select_random_subsequent_point(pred_mask=pred_mask, gt_mask=gt_mask)
        coord_collect = []
        label_collect = []
        for i in range(pred_mask.shape[0]):
            coords, label = self.point_prompt_generator.select_random_subsequent_point(pred_mask[i][0], gt_mask[i][0])
            if label == -1:
                return None, None
            coord_collect.append(coords)
            label_collect.append(label)
        
        coord_collect = np.stack(coord_collect, axis=0)
        label_collect = np.stack(label_collect, axis=0)
        return coord_collect, label_collect

    def get_noisy_box_from_box(self, box):
        # Get noisy box from labeled box
        return self.box_prompt_generator.add_random_noise_to_bbox(box)

    # def get_prompt_mask(self, )

# class ValidEngine(DataEngine):
#     def __init__(self, dataset=None, img_size=(1024,1024), is_train=False) -> None:
#         # assert dataset is not None
#         self.dataset = dataset
#         self.is_train = is_train
#         super().__init__(dataset=dataset, img_size=img_size)
#         self.expand_dataset_ratio = 1

#     # def _get_dataset(self, dirpath):
#     #     self.dataset = Dataset3D(dirpath=dirpath, is_train=self.is_train)  

#     def __len__(self):
#         return len(self.dataset)

#     def _get_true_index(self, idx):
#         return idx

class DataManager:
    def __init__(self, img_size=None) -> None:
        self.point_prompt_generator = PointPromptGenerator(img_size)
        self.box_prompt_generator = BoxPromptGenerator(img_size)

    def get_prompt_point(self, gt_mask):
        return self.point_prompt_generator.get_prompt_point(gt_mask)

    def get_prompt_box(self, gt_mask):
        return self.box_prompt_generator.get_prompt_box(gt_mask)
        
    def get_subsequent_prompt_point(self, pred_mask, gt_mask):
        # return self.point_prompt_generator.select_random_subsequent_point_torch(pred_mask, gt_mask)
        # return self.point_prompt_generator.select_random_subsequent_point(pred_mask=pred_mask, gt_mask=gt_mask)
        coord_collect = []
        label_collect = []
        for i in range(pred_mask.shape[0]):
            coords, label = self.point_prompt_generator.select_random_subsequent_point(pred_mask[i][0], gt_mask[i][0])
            if label == -1:
                return None, None
            coord_collect.append(coords)
            label_collect.append(label)
        
        coord_collect = np.stack(coord_collect, axis=0)
        label_collect = np.stack(label_collect, axis=0)
        return coord_collect, label_collect

    def get_noisy_box_from_box(self, box):
        # Get noisy box from labeled box
        return self.box_prompt_generator.add_random_noise_to_bbox(box)
    

if __name__ == "__main__":
    dataset = DataEngine()
    data = dataset.__getitem__(0)

    import ipdb; ipdb.set_trace()