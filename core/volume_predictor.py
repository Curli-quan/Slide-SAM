import torch
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
from einops import rearrange
from utils.amg import MaskData, batched_mask_to_box, batched_mask3d_to_box
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from utils.amg3d import build_point_grid, calculate_stability_score_3d, MaskData3d, batch_iterator
from utils.amg import calculate_stability_score
from utils.transforms3d import ResizeLongestSide, SimpleResize
from einops import rearrange, repeat
# from datasets.data_engine import ValidEngine, BoxPromptGenerator
from datasets.data_engine import DataEngine, DataManager, BoxPromptGenerator, PointPromptGenerator
import cv2
import torch.nn.functional as F
from tutils.new.manager import ConfigManager
from tutils.nn.data import read, itk_to_np, write, np_to_itk
from torchvision.utils import save_image
from einops import rearrange, reduce, repeat
from core.loss import compute_dice_np


class PseudoPredictor:
    # def __init__(self) -> None:
    #     self.image_encoder = 
    def predict(self, *args, **kwargs):
        mask = np.zeros((1024,1024))
        mask[:500,:500] = 1
        return mask


class VolumePredictor:
    def __init__(
        self, 
        model,
        slice_per_batch: int = 4,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 16,
        pred_iou_thresh: float = 0.5, # debug, standard value is 0.88,
        stability_score_thresh: float = 0.6, # debug, standard value is 0.95, 
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        use_postprocess = True,
        use_noise_remove = True,
        ) -> None:
        self.model = model
        self.im_size = (model.image_encoder.img_size, model.image_encoder.img_size)
        self.slice_per_batch = slice_per_batch
        self.point_grids = build_point_grid(points_per_side, self.im_size)
        self.features = None
        self.is_image_set = False
        self.transform = SimpleResize(model.image_encoder.img_size)

        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.masks3d = dict()

        self.box_prompt_generator = BoxPromptGenerator(size=(1024,1024))
        self.masks3d = None
        self.stability_score_2d = None
        self.input_size = model.image_encoder.img_size
        self.use_postprocess = use_postprocess
        self.use_noise_remove = use_noise_remove
        if not use_postprocess:
            print("Warning! No postprocess")
        if not use_noise_remove:
            print("Warning! No use_noise_remove")
        # self.original_size = (1024,1024)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
        self.masks3d = None
        self.stability_score_2d = None

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "nifti",
    ) -> None:
        # Transform the image to the form expected by the model
        self.original_size = image.shape
        input_image_torch = torch.as_tensor(image, device=self.device)
        input_image_torch = self.transform.apply_image(input_image_torch.float())
        assert np.argmin(input_image_torch.shape) == 0, f"Got image.shape: {input_image_torch.shape}"
        maxlen = input_image_torch.shape[0]
        slices = []
        for i in range(1, maxlen-1):
            slices.append(input_image_torch[i-1:i+2, :, :])
        input_slices = torch.stack(slices, axis=0)
        self.set_torch_image(input_slices)

    def batched_to_RGB(self, images):
        for i in range(images.shape[0]):
            images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min()) * 255
        return images

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
    ) -> None:
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.input_size = tuple(transformed_image.shape[-2:])
        transformed_image = self.batched_to_RGB(transformed_image)
        input_image = self.model.preprocess(transformed_image)
        features = []
        for input_image_batch in batch_iterator(self.slice_per_batch, input_image):
            # print(input_image_batch[0].shape)
            features_batch = self.model.image_encoder(input_image_batch[0]).cpu()
            features.append(features_batch)
        self.features = torch.cat(features, axis=0)
        self.is_image_set = True

    def __call__(self, x, *args: Any, **kwds: Any) -> Any:
        return self.predict_volume(x)
    
    def merge_to_mask3d(self, idx, masks:MaskData):
        if masks._stats == {} or len(masks['masks']) == 0:
            print("No masks")
            return
        if self.masks3d is None:
            self.masks3d = np.zeros(self.original_size)
        if self.stability_score_2d is None:
            self.stability_score_2d = np.zeros(self.original_size[0])
        masks_values = masks['masks']
        for mask_value in zip(masks_values):
            old_mask = self.masks3d[idx-1:idx+2]
            # self.masks3d[idx-1:idx+2] = np.where(mask_value > old_mask, mask_value, old_mask)
            self.masks3d[idx-1:idx+2] = mask_value + old_mask

        self.stability_score_2d[idx] = masks['stability_score_2d'][0,0]

    def postprocess_3d(self, masks3d):
        # add removing noise ?
        return masks3d > 0

    def _debug_predict_slice(
        self,
        x,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        template_slice_id:int = None,
        return_stability: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            main entrence
                predict 3d volume

            x: volume: (c h w)
            box: [[x,y,x,y]]
        """
        # Check Input
        assert len(x.shape) == 3
        assert box is None or len(box.shape) == 2

        # preprocess
        # x = np.clip(x, -200,400)
        print(f"Checking Data range: [{x.min()}, {x.max()}]" )

        # Adjust direction
        indicator = np.argmin(x.shape)
        if indicator == 0:
            pass
        elif indicator == 1:
            x = rearrange(x, "h c w -> c h w")
        elif indicator == 2:
            x = rearrange(x, "h w c -> c h w")
        else:
            raise NotImplementedError

        # Preprocess prompts
        self.original_size = x.shape[1:]
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        # set 3d image
        self.set_image(x)
        
        # predict center slice
        center_idx = template_slice_id if template_slice_id is not None else x.shape[0] // 2
        # print("Processing ", center_idx)
        center_masks = self._predict_center_slice(center_idx, point_coords, box)
        return center_masks['masks']

    def predict_volume(
        self,
        x,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        template_slice_id:int = None,
        return_stability: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            main entrence
                predict 3d volume

            x: volume: (c h w)
            box: [[x,y,x,y]]
        """
        # Check Input
        assert len(x.shape) == 3
        assert box is None or len(box.shape) == 2

        # preprocess
        # x = np.clip(x, -200,400)
        print(f"Checking Data range: [{x.min()}, {x.max()}]" )

        # Adjust direction
        indicator = np.argmin(x.shape)
        if indicator == 0:
            pass
        elif indicator == 1:
            x = rearrange(x, "h c w -> c h w")
        elif indicator == 2:
            x = rearrange(x, "h w c -> c h w")
        else:
            raise NotImplementedError

        # Preprocess prompts
        self.original_size = x.shape[1:]
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        # set 3d image
        self.set_image(x)
        
        # predict center slice
        center_idx = template_slice_id if template_slice_id is not None else x.shape[0] // 2
        # print("Processing ", center_idx)
        center_masks = self._predict_center_slice(center_idx, point_coords, box)
        if center_masks._stats == {}:
            print("Ends for no mask.")
            raise ValueError
        self.merge_to_mask3d(center_idx, center_masks)

        previous_masks = center_masks
        for i in range(center_idx+1, x.shape[0]-1):
            # print("Processing downward", i)
            previous_masks = self._predict_slice(i, previous_masks, orientation="down")
            if previous_masks._stats == {}:
                print("Ends for no mask.")
                break
            self.merge_to_mask3d(i, previous_masks)

        previous_masks = center_masks
        for i in np.arange(1, center_idx)[::-1]:
            # print("Processing upward", i)
            previous_masks = self._predict_slice(i, previous_masks, orientation="up")
            if previous_masks._stats == {}:
                print("Ends for no mask.")
                break
            self.merge_to_mask3d(i, previous_masks)
        
        if self.masks3d is None:
            self.masks3d = np.zeros_like(x)
        if return_stability:
            return self.postprocess_3d(self.masks3d), self.stability_score_2d
        return self.postprocess_3d(self.masks3d)

    def _predict_center_slice(self, idx, point_prompt=None, box_prompt=None):
        if box_prompt is not None:
            masks = self.genetate_masks_from_boxes(idx, all_boxes=box_prompt, tags=["center_slice"])
            masks.to_numpy()
            return masks
        if point_prompt is not None:
            masks = self.genetate_masks_from_point_grids(idx, point_prompt)
            masks.to_numpy()
            return masks
        raise ValueError("No prompts! ?")
    
    def _predict_slice(self, idx, previous_masks, orientation):
        scaled_boxes, tags = self.generate_prompts_from_previous_masks(previous_masks, orientation)
        masks = self.genetate_masks_from_boxes(idx, all_boxes=scaled_boxes, tags=tags)
        masks.to_numpy()
        return masks

    def generate_prompts_from_previous_masks(self, previous_masks: MaskData, orientation):
        if orientation == "down":
            masks = previous_masks['masks'][:,2,:,:]
        elif orientation == "up":
            masks = previous_masks['masks'][:,0,:,:]
        else:
            raise ValueError
        raw_tags = previous_masks['tags']

        scaled_boxes = []
        tags = []
        for mask, tag in zip(masks, raw_tags):
            if mask.sum() <= 50:
                continue
            # mask = self.remove_mask_noise(mask)
            # if mask.sum() <= 50:
            #     continue
            mask = F.interpolate(torch.Tensor(mask).float()[None,None,:,:], self.input_size).squeeze().numpy()
            # scaled_boxes.append(self.box_prompt_generator.mask_to_bbox(mask))
            scaled_boxes.append(self.box_prompt_generator.enlarge(self.box_prompt_generator.mask_to_bbox(mask)))
            tags.append(tag)
        scaled_boxes = np.array(scaled_boxes)
        return scaled_boxes, tags

    def genetate_masks_from_point_grids(self, idx, points_for_image):
        idx = idx - 1 # ignore the head and tail slices
        # Get points for this crop
        data = MaskData()
        tags = [f"s{idx}_p{p}" for p in range(points_for_image.shape[0])]
        for (points, batched_tags) in batch_iterator(self.points_per_batch, points_for_image, tags):
            batch_data = self._process_batch(idx, points=points, tags=batched_tags)
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
        # keep_by_nms = batched_nms(
        #     data["boxes"].float(),
        #     data["iou_preds"],
        #     torch.zeros_like(data["boxes"][:, 0]),  # categories
        #     iou_threshold=self.box_nms_thresh,
        # )
        # data.filter(keep_by_nms)
        return data

    def genetate_masks_from_boxes(self, idx, all_boxes, tags):
        idx = idx - 1
        data = MaskData()
        for (batched_boxes, batched_tags) in batch_iterator(self.points_per_batch, all_boxes, tags):
            batch_data = self._process_batch(idx, boxes=batched_boxes, tags=batched_tags)
            data.cat(batch_data)
            del batch_data
        return data

    def _process_batch(
            self, 
            fea_slice_idx, 
            points=None, 
            boxes=None, 
            multimask_output=True, 
            tags=None
        ) -> MaskData:
        """
            Process with a subset of points. (bacause so many points can not be feed in one time)
        """
        if points is not None:
            in_points = torch.as_tensor(points, device=self.model.device)
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            masks, iou_preds, _ = self.predict_torch_by_sliceidx(
                fea_slice_idx=fea_slice_idx,
                point_coords=in_points[:, None, :],
                point_labels=in_labels[:, None],
                multimask_output=multimask_output,
                return_logits=True,
            )
            masks = rearrange(masks, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
        elif boxes is not None:
            # in_points = torch.as_tensor(points, device=self.model.device)
            # in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            boxes = torch.as_tensor(boxes, device=self.model.device)
            masks, iou_preds, _ = self.predict_torch_by_sliceidx(
                fea_slice_idx=fea_slice_idx,
                boxes=boxes,
                multimask_output=multimask_output,
                return_logits=True,
            )
            masks = rearrange(masks, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
        else:
            raise ValueError(f"No points or boxes")

        indices = iou_preds.argmax(axis=1)
        # indices = torch.tensor([2], device=iou_preds.device)
        pred_maxiou = []
        for pred, i in zip(masks, indices):
            pred_maxiou.append(pred[i,:,:,:])
        masks = torch.stack(pred_maxiou, axis=0)

        iou_maxiou = []
        for iou, i in zip(iou_preds, indices):
            iou_maxiou.append(iou[i])
        iou_preds = torch.stack(iou_maxiou)

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.detach().cpu(),
            iou_preds=iou_preds.detach().cpu(),
            # points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
            # tags=repeat(np.array(tags), "c -> (c 3)")
            tags=np.array(tags),
        )
        del masks

        # Filter by area
        # if True:
        #     keep_mask = (data['masks']>0).sum(-1).sum(-1).sum(-1) < (data['masks']<=0).sum(-1).sum(-1).sum(-1) * 0.4
        #     data.filter(keep_mask)
            # print("keep mask / pred", keep_mask.sum())
        
        # Filter Background
        # if True:
        #     keep_mask = 

        # Filter by predicted IoU
        if self.use_postprocess:
            if self.pred_iou_thresh > -0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                # print("pred_iou", data["iou_preds"], (data["masks"]>0).sum(-1).sum(-1))
                data.filter(keep_mask)
                # print("keep mask / pred", keep_mask.sum())

            # Calculate stability score
            data["stability_score"] = calculate_stability_score_3d(
                data["masks"], self.model.mask_threshold, self.stability_score_offset
            ) # .mean(axis=-1)
            if self.stability_score_thresh > 0.0:
                # print("stability", data["stability_score"], (data["masks"]>0).sum(-1).sum(-1))
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
                # print("keep mask / stable", keep_mask.sum())

        data["stability_score_2d"] = calculate_stability_score(
            data["masks"][:,1:2,:,:], self.model.mask_threshold, self.stability_score_offset
        )
        
        # Threshold masks and calculate boxes
        data['logits'] = data['masks']
        data["noisy_masks"] = data["logits"] > self.model.mask_threshold
        
        # data['masks'] = torch.zeros_like(data['noisy_masks'], dtype=data['noisy_masks'].dtype, device=data['noisy_masks'].device)
        b, c,_,_ = data["noisy_masks"].shape
        data['masks'] = data["noisy_masks"].float()

        if self.use_noise_remove:
            for i in range(b):
                for j in range(c):
                    data['masks'][i,j,:,:] = torch.Tensor(self.remove_mask_noise(data['noisy_masks'][i,j,:,:]))

        # data["boxes"] = batched_mask_to_box(reduce(data["masks"], "b c h w -> b h w", reduction="sum")>0)
        data["boxes"] = batched_mask_to_box(data["masks"][:,1,:,:]>0)
        return data

    # @staticmethod
    # def calculate_


    @staticmethod
    def batched_remove_noise(masks):
        ori_shape = masks.shape

    def remove_mask_noise(self, mask):
        # mask_sum = mask.sum()
        kerner_size = min(mask.sum() // 20, 8)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        kernel = np.ones((kerner_size,kerner_size), dtype=np.uint8)
        opening = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, 1)
        return opening
    
    @torch.no_grad()
    def predict_torch_by_sliceidx(
        self,
        fea_slice_idx: int,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features[fea_slice_idx].to(self.model.device),
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size[1:])

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
    
    def valid_box(self, data, batch_idx):
        # valid image with box, or point prompt
        assert data['img'].shape[0] == 1, f"shape {data['img'].shape}"
        image = data['img']
        label = data['label']

        box = BoxPromptGenerator().mask_to_bbox(label)
        box_mask3d = self.predict_volume(
            x=image,
            box=box,
        )
        dice = compute_dice_np(box_mask3d, label.detach().cpu().numpy())
        


if __name__ == "__main__":
    from core.learner3 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry
    from trans_utils.data_utils import Data3dSolver

    config = ConfigManager()
    config.add_config("configs/vit_b_103.yaml")

    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=None)
    learner = SamLearner(sam_model=sam, config=config, data_engine=DataManager(img_size=(1024,1024)))
    learner.use_lora()
    pth = "model_iter_360000.pth"
    learner.load_well_trained_model(pth)
    learner.cuda()

    predictor = VolumePredictor(
        model=learner.model,
        use_postprocess=True,
        use_noise_remove=True,)

    # Load data
    img_path = "/home1/quanquan/datasets/07_WORD/WORD-V0.1.0/imagesVa/word_0001.nii.gz" # "/home1/quanquan/datasets/59_SABS/sabs_CT_normalized/image_5.nii.gz"
    label_path = "/home1/quanquan/datasets/07_WORD/WORD-V0.1.0/labelsVa/word_0001.nii.gz"
    volume = itk_to_np(read(img_path))  # test several slices
    label_itk = read(label_path)
    spacing = label_itk.GetSpacing()
    label = itk_to_np(label_itk) == 1
    volume = np.clip(volume, -200, 400)
    
    # Select the slice with the largest mask
    s = reduce(label, "c h w -> c", reduction="sum")
    coords = np.nonzero(s)
    x_min = np.min(coords[0])
    x_max = np.max(coords[0])
    template_slice_id = s.argmax()

    box = BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id])
    box = np.array([box])

    pred = predictor.predict_volume(
            x=volume,
            box=box,            
            template_slice_id=template_slice_id,
            return_stability=False,
        )

    Data3dSolver().simple_write(pred, path="mask.nii.gz", spacing=spacing)
    Data3dSolver().simple_write(label, path="gt.nii.gz", spacing=spacing)