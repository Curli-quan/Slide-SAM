import torch
import torchvision
import numpy as np
from tutils.trainer import Trainer, LearnerModule
from torch.utils.data import DataLoader
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange, repeat
from torch.nn import functional as F
import os
from typing import Optional, Tuple
import torch.optim.lr_scheduler as lr_scheduler

from modeling.sam3d import Sam
# from segment_anything.utils.transforms import ResizeLongestSide
from utils.transforms import ResizeLongestSide
from .loss import compute_all_loss, ranked_combined_loss, compute_iou, combined_loss
from .lora_sam import LoRA_Sam
from safetensors import safe_open
from datasets.data_engine import DataEngine


# def lr_schedule(epoch):
#     if epoch < 250:
#         return (epoch + 1) / 250 * 0.0008 + 0.00004
#     elif epoch < 500:
#         return 0.0001
#     else:
#         return 0.0001

def lr_schedule(epoch):
    if epoch < 250:
        return (epoch + 1) / 250 * 0.1
    elif epoch < 500:
        return 0.01
    else:
        return 0.001

class SamLearner(LearnerModule):
    def __init__(
        self,
        sam_model: Sam,
        config=None, 
        logger=None, 
        data_engine=DataEngine(None, img_size=(1024,1024)), 
        lora_module=None,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.config = config
        self.logger = logger
        self.model = sam_model
        self.net = self.model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()
        self.data_engine = data_engine
        self.features = None
        self.lora_module = lora_module

    def save(self, pth, *args, **kwargs):
        # Default: "/model_epoch_{}.pth".format(epoch)
        torch.save(self.net.state_dict(), pth)
        lora_path = pth.replace(".pth", "_lora.safetensors")
        self.lora_module.save_lora_parameters(lora_path)
        return True

    def load_pretrained_model(self, pth, *args, **kwargs):        
        """
            Unmatched: prompt_encoder.mask_downscaling.0.weight
                their: torch.Size([4, 1, 2, 2])
                our: torch.Size([4, 3, 2, 2])
            Unmatched: mask_decoder.mask_tokens.weight
                their: torch.Size([4, 256])
                our: torch.Size([12, 256])

        """
        state_dict = torch.load(pth)
        model_state_dict = self.model.state_dict()
        model_state_dict.update(state_dict)
        model_state_dict['prompt_encoder.mask_downscaling.0.weight'] = repeat(state_dict['prompt_encoder.mask_downscaling.0.weight'], "a 1 c d -> a b c d", b=3)
        model_state_dict['mask_decoder.mask_tokens.weight'] = repeat(state_dict['mask_decoder.mask_tokens.weight'], "a d -> (a 3) d")
        hyper_params_names = [k for k in model_state_dict.keys() if k.startswith("mask_decoder.output_hypernetworks_mlps")]
        for name in hyper_params_names:
            words = name.split('.')
            words[2] = str(int(words[2]) // 3)
            name_to_copy = ".".join(words)
            model_state_dict[name] = state_dict[name_to_copy]
        # for k, v in state_dict.items():
        #     if model_state_dict[k].shape != state_dict[k].shape:
        #         print("Unmatched:", k)
        self.model.load_state_dict(model_state_dict)

    def load_well_trained_model(self, pth=None):
        pth = self.config['training']['breakpoint_path'] + "/ckpt_v/model_latest.pth" if pth is None else pth
        print("Loading from ", pth)
        state_dict = torch.load(pth, map_location="cpu")
        # print(state_dict.keys())
        # for k in state_dict.keys():
        #     print(k)
        # exit(0)
        self.model.load_state_dict(state_dict)
        # self.lora_module.load_lora_parameters(pth.replace(".pth", "_lora.safetensors"))

    def use_lora(self, r=8):        
        lora_r = r
        lora_sam = LoRA_Sam(self.model, lora_r, freeze_prompt_encoder=True)
        self.lora_module = lora_sam

    def configure_optimizers(self, **kwargs):
        optimizer = optim.AdamW(params=self.model.parameters(), \
                           lr=self.config['training']['lr'], betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=self.config['training']['weight_decay'])
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule)
        scheduler = None
        return {'optimizer': optimizer, "scheduler": scheduler}

    def load_optim(self, optimizer, pth=None, *args):
        pth = self.config['training']['breakpoint_path'] + "/ckpt/optim_latest.pth"
        print("Load Optimizer from ", pth)
        state_dict = torch.load(pth)
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict.get('epoch', 0) + 1
        return start_epoch

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        gt_mask = data['label']        
        prompt_point = data['prompt_point'] # shape: (b, 2)
        batch_size = prompt_point.shape[0]
        point_label = torch.ones((batch_size, 1)) #.to(prompt_point.device)
        prompt_box = data['prompt_box']

        prompt_point = rearrange(prompt_point, "b c -> b 1 c")
        prompt_box = rearrange(prompt_box, "b c -> b 1 c")
        assert img.shape[1:] == (3,1024,1024),f"{__file__} Got{img.shape}"
        assert prompt_point.shape[1:] == (1,2), f"{__file__} Got{prompt_point.shape}"
        assert point_label.shape[1:] == (1,), f"{__file__} Got{point_label.shape}"
        assert prompt_box.shape[1:] == (1,4), f"{__file__} Got{prompt_box.shape}"

        self.set_torch_image(img, img.shape[2:])
        if np.random.random() > 0.5:
            pred_masks, iou_predictions, logits = self.predict_torch(
                point_coords=prompt_point,
                point_labels=point_label,  
                multimask_output=True,   
                return_logits=True,       
            )
        else:            
            pred_masks, iou_predictions, logits = self.predict_torch(
                point_coords=None,
                point_labels=None,  
                boxes=prompt_box,
                multimask_output=True,  
                return_logits=True,              
            )
        # assert pred_masks.shape == gt_mask.shape, f"Got {pred_masks.shape}, {gt_mask.shape}"
        loss_1, fl, dl = ranked_combined_loss(pred_mask=pred_masks, gt_mask=gt_mask, iou_pred=iou_predictions)

        # print("Debug trainer: 2", prompt_point.shape, point_label.shape, prompt_box.shape)
        # Stage 2: based on the above, add more points as prompts
        return {"loss": loss_1, "fl": fl.mean(), "dl": dl.mean()}
    
    # @torch.no_grad()
    def generate(self, image, prompt_point):        
        orig_size = image.shape[2:]
        assert image.shape[1:] == (3,1024,1024),f"{__file__} Got{image.shape}"
        if not self.is_image_set:
            self.set_torch_image(image, orig_size)

        assert prompt_point.shape[1:] == (1,2), f"{__file__} Got{prompt_point.shape}"
        # assert point_label.shape[1:] == (1,), f"{__file__} Got{point_label.shape}"
        point_label = torch.ones(prompt_point.size()[:-1])
        pred_masks, scores, logits = self.predict_torch(
            point_coords=prompt_point,
            point_labels=point_label,  
            mask_input=None,
            multimask_output=True, 
        )
        return pred_masks
    
    # @torch.no_grad()
    def generate_by_box(self, image, prompt_box):        
        orig_size = image.shape[2:]
        assert image.shape[1:] == (3,1024,1024),f"{__file__} Got{image.shape}"
        if not self.is_image_set:
            self.set_torch_image(image, orig_size)

        assert prompt_box.shape[1:] == (1,4), f"{__file__} Got{prompt_box.shape}"
        pred_masks, scores, logits = self.predict_torch(
            point_coords=None,
            point_labels=None,  
            boxes=prompt_box,
            mask_input=None,
            multimask_output=True, 
        )
        return pred_masks

    @staticmethod
    def select_best_mask(predictions, ground_truth):
        # Move tensors to the same device (if not already on the same device)
        # if predictions.device != ground_truth.device:
        #     predictions = predictions.to(ground_truth.device)

        # Compute IoU between each prediction and ground truth
        if predictions.shape[1] == 9:
            predictions = rearrange(predictions, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
            ground_truth = repeat(ground_truth, "b d h w -> b c d h w", c=3)
        else:            
            predictions = rearrange(predictions, "b d h w -> b 1 d h w")
            ground_truth = rearrange(ground_truth, "b d h w -> b 1 d h w")
        intersection = torch.sum(predictions * ground_truth, dim=(-3, -2, -1))
        union = torch.sum(predictions + ground_truth, dim=(-3, -2, -1)) - intersection
        iou = intersection / (union + 1e-6)

        # Select the prediction with maximum IoU for each image in the batch
        best_indices = torch.argmax(iou, dim=1)
        best_masks = torch.gather(predictions, 1, best_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, predictions.shape[-3], predictions.shape[-2], predictions.shape[-1]))

        return best_masks

    # ===============================================
    def predict_multi_prompt(
        self, 
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        mask_logits: Optional[torch.Tensor],  
    ):
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        
        if point_coords is not None:
            points = (coords_torch, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_logits,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        return masks, iou_predictions, low_res_masks

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    # @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
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

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        # masks = masks[0].detach().cpu().numpy()
        # iou_predictions = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks = low_res_masks[0].detach().cpu().numpy()
        return masks, iou_predictions, low_res_masks

    # @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
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

        sparse_embeddings, dense_embeddings = self._get_prompt_embedding(points, boxes, mask_input)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        # import ipdb; ipdb.set_trace()

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
    
    # @torch.no_grad()
    def _get_prompt_embedding(self, points, boxes, mask_input):
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        return sparse_embeddings, dense_embeddings


    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

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
 
    
    def validation_step(self, data, batch_idx=0, **kwargs):
        img = data['img']
        gt_mask = data['label']        
        prompt_point = data['prompt_point'] # shape: (b, 2)
        batch_size = prompt_point.shape[0]
        point_label = torch.ones((batch_size, 1)) #.to(prompt_point.device)
        prompt_box = data['prompt_box']
        gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=3)

        prompt_point = rearrange(prompt_point, "b c -> b 1 c")
        prompt_box = rearrange(prompt_box, "b c -> b 1 c")
        assert img.shape[1:] == (3,1024,1024),f"{__file__} Got{img.shape}"
        assert prompt_point.shape[1:] == (1,2), f"{__file__} Got{prompt_point.shape}"
        assert point_label.shape[1:] == (1,), f"{__file__} Got{point_label.shape}"
        assert prompt_box.shape[1:] == (1,4), f"{__file__} Got{prompt_box.shape}"

        self.set_torch_image(img, img.shape[2:])

        # Stage 1: use the 1st prompt, box or point
        iou_details = {}
        pred_masks1, iou_predictions1, logits1 = self.predict_torch(
            point_coords=prompt_point,
            point_labels=point_label,  
            multimask_output=True,   
            return_logits=True,       
        )
        
        pred_masks1 = rearrange(pred_masks1, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
        loss_point, fl, dl, _, _ = ranked_combined_loss(pred_mask=pred_masks1, gt_mask=gt_mask, iou_pred=iou_predictions1)
        iou_details['loss_point'] = loss_point.mean()
        iou_details['loss_point_fl'] = fl.mean()
        iou_details['loss_point_dl'] = dl.mean() 

        iou = compute_iou((pred_masks1>0).float(), gt_mask)
        iou, _ = torch.max(iou, axis=1)
        iou_details['iou_point'] = iou.mean()

        pred_masks2, iou_predictions2, logits2 = self.predict_torch(
            point_coords=None,
            point_labels=None,  
            boxes=prompt_box,
            multimask_output=True,  
            return_logits=True,              
        )        
        pred_masks2 = rearrange(pred_masks2, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
        loss_box, fl, dl, _, _ = ranked_combined_loss(pred_mask=pred_masks2, gt_mask=gt_mask, iou_pred=iou_predictions2)
        iou_details['loss_box'] = loss_box.mean()
        iou_details['loss_box_fl'] = fl.mean()
        iou_details['loss_box_dl'] = dl.mean()

        iou = compute_iou((pred_masks2>0).float(), gt_mask)
        iou, _ = torch.max(iou, axis=1)
        iou_details['iou_box'] = iou.mean()    

        # import ipdb; ipdb.set_trace()


        # gt_mask_np = gt_mask.detach().cpu().numpy()
        # for step in range(8):
        #     continue
        #     # n
        #     best_pred_masks = self.select_best_mask(pred_masks, gt_mask)            
        #     best_pred_masks_np = best_pred_masks.detach().cpu().numpy()
            
        #     # import ipdb; ipdb.set_trace()
        #     mask_input = logits[0, np.argmax(scores[0].detach().cpu().numpy()), :, :]  # Choose the model's best mask

        #     sub_points, sub_labels = self.data_engine.get_subsequent_prompt_point(best_pred_masks_np, gt_mask_np)
        #     # sub_points, sub_labels = self.data_engine.point_prompt_generator.select_random_subsequent_point(best_pred_masks_np[0][0], gt_mask_np[0][0])
            
        #     y, x = sub_points[0][1], sub_points[0][0]
        #     assert gt_mask_np[0][0][y,x] + best_pred_masks_np[0][0][y,x] == 1, f"{__file__} Got{gt_mask_np[0][0][y,x], best_pred_masks_np[0][0][y,x]}"
        #     assert gt_mask_np[0][0][y,x] == sub_labels, f"{__file__} Got{ gt_mask_np[0][0][y,x]}, {sub_labels}"
        #     assert best_pred_masks_np[0][0][y,x] == (1-sub_labels), f"{__file__} Got{ gt_mask_np[0][0][y,x]}, {1-sub_labels}"
        #     # import ipdb; ipdb.set_trace()
        #     # assert sub_points

        #     # sub_points = np.array(sub_points)[None,...].astype(int) 
        #     # sub_labels = np.array(sub_labels)[None,...] 
        #     prompt_point = np.concatenate([prompt_point, sub_points], axis=0)
        #     point_label = np.concatenate([point_label, sub_labels], axis=0)

        #     # import ipdb; ipdb.set_trace()

        #     pred_masks2, scores, logits = model.predict(
        #         point_coords=prompt_point,
        #         point_labels=point_label,  
        #         mask_input=mask_input[None,...],
        #         multimask_output=False, 
        #     )
            
        #     iou = compute_iou(pred_masks2, gt_mask)
        #     iou, _ = torch.max(iou, axis=1)
        #     iou_details[f'point_{step+2}'] = iou

        return iou_details
   