import torch
import torchvision
import numpy as np
from tutils.trainer import Trainer, LearnerModule
from einops import rearrange, repeat, reduce
import torch.optim.lr_scheduler as lr_scheduler

from core.loss import ranked_combined_loss_with_indicators
from .learner2 import SamLearner as basic_learner
from .loss import compute_all_loss, ranked_combined_loss, compute_iou, combined_loss


class SamLearner(basic_learner):
    
    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        gt_mask = data['label']        
        prompt_point = data['prompt_point'] # shape: (b, 2)
        batch_size = prompt_point.shape[0]
        point_label = torch.ones((batch_size, 1)) #.to(prompt_point.device)
        prompt_box = data['prompt_box']
        indicators = data['indicators']
        # print(data['name'])

        prompt_point = rearrange(prompt_point, "b c -> b 1 c")
        prompt_box = rearrange(prompt_box, "b c -> b 1 c")
        assert img.shape[1:] == (3,1024,1024),f"{__file__} Got{img.shape}"
        assert prompt_point.shape[1:] == (1,2), f"{__file__} Got{prompt_point.shape}"
        assert point_label.shape[1:] == (1,), f"{__file__} Got{point_label.shape}"
        assert prompt_box.shape[1:] == (1,4), f"{__file__} Got{prompt_box.shape}"

        self.set_torch_image(img, img.shape[2:])
        # if np.random.random() > 0.5:
        pred_masks, iou_predictions, logits = self.predict_torch(
            point_coords=prompt_point,
            point_labels=point_label,  
            multimask_output=True,   
            return_logits=True,       
        )
        loss_1, fl, dl, min_losses, selected_losses, _ = ranked_combined_loss_with_indicators(pred_mask=pred_masks, gt_mask=gt_mask, iou_pred=iou_predictions, indicators=indicators)
        # else:            
        pred_masks, iou_predictions, logits = self.predict_torch(
            point_coords=None,
            point_labels=None,  
            boxes=prompt_box,
            multimask_output=True,  
            return_logits=True,              
        )
        # assert pred_masks.shape == gt_mask.shape, f"Got {pred_masks.shape}, {gt_mask.shape}"
        loss_2, fl, dl, min_losses, selected_losses, _ = ranked_combined_loss_with_indicators(pred_mask=pred_masks, gt_mask=gt_mask, iou_pred=iou_predictions, indicators=indicators)

        loss = loss_1 + loss_2
        if loss < -999 or torch.isnan(loss):
            print("Warning! Loss Error! ")
            print(data['name'])

        # print("Debug trainer: 2", prompt_point.shape, point_label.shape, prompt_box.shape)
        # Stage 2: based on the above, add more points as prompts
        return {"loss": loss_1 + loss_2, "point_loss": loss_1.mean(), "box_loss": loss_2, "fl": fl.mean(), "dl": dl.mean(), "min_losses": min_losses, "selected_losses": selected_losses}
    
    
    def validation_step(self, data, batch_idx=0, **kwargs):
        img = data['img']
        gt_mask = data['label']        
        prompt_point = data['prompt_point'] # shape: (b, 2)
        batch_size = prompt_point.shape[0]
        point_label = torch.ones((batch_size, 1)) #.to(prompt_point.device)
        prompt_box = data['prompt_box']
        indicators = data['indicators']
        print(data['name'])
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

        if len(pred_masks1.shape) == 4:
            pred_masks1 = rearrange(pred_masks1, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
        
        loss_point, fl, dl, min_losses, selected_losses, min_indices = ranked_combined_loss_with_indicators(pred_mask=pred_masks1, gt_mask=gt_mask, iou_pred=iou_predictions1, indicators=indicators)
        iou_details['loss_point'] = loss_point.mean()
        iou_details['loss_point_fl'] = fl.mean()
        iou_details['loss_point_dl'] = dl.mean() 
        
        if len(gt_mask.shape) == 4:
            gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=3)

        indices = iou_predictions1.argmax(axis=1)
        pred_maxiou = []
        for pred, i in zip(pred_masks1, indices):
            pred_maxiou.append(pred[i,:,:,:])
        pred_maxiou = torch.stack(pred_maxiou, axis=0)
        iou = compute_iou2(torch.tensor(pred_maxiou>0, dtype=gt_mask.dtype), gt_mask[:,0,:,:,:]).detach()
        iou_details['iou_point'] = iou.mean()

        iou = compute_iou(torch.tensor(pred_masks1>0, dtype=gt_mask.dtype), gt_mask).detach()
        iou, _ = torch.max(iou, axis=1)
        iou_details['iou_point_max'] = iou.mean()

        pred_masks2, iou_predictions2, logits2 = self.predict_torch(
            point_coords=None,
            point_labels=None,  
            boxes=prompt_box,
            multimask_output=True,  
            return_logits=True,              
        )        
        loss_box, fl, dl, min_losses, selected_losses, min_indices = ranked_combined_loss_with_indicators(pred_mask=pred_masks2, gt_mask=gt_mask, iou_pred=iou_predictions2, indicators=indicators)
        iou_details['loss_box'] = loss_box.mean()
        iou_details['loss_box_fl'] = fl.mean()
        iou_details['loss_box_dl'] = dl.mean()

        if len(gt_mask.shape) == 4:
            gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=3)
        if len(pred_masks2.shape) == 4:
            pred_masks2 = rearrange(pred_masks2, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
        
        indices = iou_predictions2.argmax(axis=1)
        pred_maxiou = []
        for pred, i in zip(pred_masks2, indices):
            pred_maxiou.append(pred[i,:,:,:])
        pred_maxiou = torch.stack(pred_maxiou, axis=0)
        iou = compute_iou2(torch.tensor(pred_maxiou>0, dtype=gt_mask.dtype), gt_mask[:,0,:,:,:]).detach()
        iou_details['iou_box'] = iou.mean()

        iou = compute_iou(torch.tensor(pred_masks2>0, dtype=gt_mask.dtype), gt_mask).detach()
        iou, _ = torch.max(iou, axis=1)
        iou_details['iou_box_max'] = iou.mean()    
        return iou_details
    

def compute_iou2(pred_mask, gt_mask):
    dtype = pred_mask.dtype
    intersection = torch.logical_and(pred_mask, gt_mask)
    intersection = reduce(intersection, "b d h w -> b", reduction='sum')
    union = torch.logical_or(pred_mask, gt_mask)
    union = reduce(union, "b d h w -> b", reduction='sum') + 1e-8
    iou = intersection / union # if union > 0 else 0
    iou = torch.tensor(iou, dtype=dtype)
    # print("ranked_combined_loss: compute_iou ", intersection.dtype, union.dtype, iou.dtype)
    return iou

# def save(img, mask, mask2):
