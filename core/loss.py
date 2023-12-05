import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat, rearrange, reduce
import numpy as np


def compute_dice_np(pred_mask, gt_mask):
    """ numpy values 
    """
    pred_mask = np.array(pred_mask>0)
    gt_mask = np.array(gt_mask>0)
    intersection = np.array(pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = intersection * 2 / union # if union > 0 else 0
    return dice


def combined_loss(logits, targets, alpha=0.2, gamma=2.0, smooth=1e-5, reduction='mean'):
    # Calculate the focal loss
    fl = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-fl)
    focal_loss = alpha * (1 - pt) ** gamma * fl

    if reduction == 'mean':
        fl = torch.mean(focal_loss)
    elif reduction == 'sum':
        fl = torch.sum(focal_loss)

    # Calculate the Dice loss
    prob = torch.sigmoid(logits)
    intersection = torch.sum(prob * targets, dim=(-2, -1))
    union = torch.sum(prob + targets, dim=(-2, -1))
    dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

    return focal_loss, dice_loss
    
    if reduction == 'mean':
        dl = torch.mean(dice_loss)
    elif reduction == 'sum':
        dl = torch.sum(dice_loss)

    # Combine the losses using the specified ratio
    loss = 20 * fl + dl
    return loss

# Assuming your prediction and ground truth tensors are named `pred` and `gt`, respectively
def mse_loss(pred, gt):
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(pred, gt)
    return loss

def compute_iou(pred_mask, gt_mask):
    dtype = pred_mask.dtype
    intersection = torch.logical_and(pred_mask, gt_mask)
    intersection = reduce(intersection, "b c d h w -> b c", reduction='sum')
    union = torch.logical_or(pred_mask, gt_mask)
    union = reduce(union, "b c d h w -> b c", reduction='sum') + 1e-8
    iou = intersection / union # if union > 0 else 0
    iou = torch.tensor(iou, dtype=dtype)
    # print("ranked_combined_loss: compute_iou ", intersection.dtype, union.dtype, iou.dtype)
    return iou

def ranked_combined_loss(pred_mask, gt_mask, iou_pred):
    # (b c1 c2 h w), c1: num_prediction;  c2: num_slices 
    if len(gt_mask.shape) == 4:
        gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=3)
    if len(pred_mask.shape) == 4:
        pred_mask = rearrange(pred_mask, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
    fl, dl = combined_loss(pred_mask, gt_mask)
    fl = reduce(fl, "b c d h w -> b c", reduction="mean")
    dl = reduce(dl, "b c d-> b c", reduction="mean")
    segment_loss = 20*fl + dl
    min_losses, min_loss_indices = torch.min(segment_loss, dim=1)
    iou = compute_iou(torch.tensor(torch.tensor(pred_mask>0, dtype=gt_mask.dtype)>0, dtype=gt_mask.dtype), gt_mask).detach().detach()
    # print("ranked_combined_loss ", iou.dtype)
    iou_loss = mse_loss(iou_pred, iou)

    selected_losses = torch.gather(iou_loss, 1, min_loss_indices.unsqueeze(1))
    selected_fl = torch.gather(fl, 1, min_loss_indices.unsqueeze(1))
    selected_dl = torch.gather(dl, 1, min_loss_indices.unsqueeze(1))
    # print(min_losses.shape, selected_losses.shape)

    total_loss = min_losses.mean() + selected_losses.mean()
    # return total_loss, min_losses, selected_losses
    return total_loss, selected_fl, selected_dl, min_losses.mean(), selected_losses.mean(), min_loss_indices

def ranked_combined_loss_one_slice(pred_mask, gt_mask, iou_pred, mask_loc):
    if len(gt_mask.shape) == 4:
        # assert gt_mask.shape[1] == 1, f"Got {gt_mask.shape}"
        gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=3)
    if len(pred_mask.shape) == 4:
        pred_mask = rearrange(pred_mask, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
    gt_mask = gt_mask[:,:,mask_loc,:,:]
    pred_mask = pred_mask[:,:,mask_loc,:,:]
    assert len(pred_mask.shape) == 5
    return ranked_combined_loss(pred_mask, gt_mask, iou_pred)   

def ranked_combined_loss_with_indicators(pred_mask, gt_mask, iou_pred, indicators):
    # indicators: indicate which slice are with the mask
    # (b c1 c2 h w), c1: num_prediction;  c2: num_slices 
    if len(gt_mask.shape) == 4:
        gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=3)
    if len(pred_mask.shape) == 4:
        pred_mask = rearrange(pred_mask, "b (c1 c2) h w -> b c1 c2 h w", c1=3, c2=3)
    
    b, c1, c2, h, w = pred_mask.shape 
    indicators = torch.tensor(indicators, dtype=pred_mask.dtype)
    indicators = repeat(indicators, "b d -> b c d h w", c=3, h=h, w=w)
    pred_mask = pred_mask * indicators
    gt_mask = gt_mask * indicators
    
    # Same as "ranked_combined_loss"
    return ranked_combined_loss(pred_mask, gt_mask, iou_pred)


def compute_all_loss_with_indicators(pred_mask, gt_mask, iou_pred, indicators):
    # indicators: indicate which slice are with the mask
    # (b c1 c2 h w), c1: num_prediction;  c2: num_slices 
    if len(gt_mask.shape) == 4:
        gt_mask = repeat(gt_mask, "b d h w -> b c d h w", c=1)
    if len(pred_mask.shape) == 4:
        pred_mask = rearrange(pred_mask, "b (c1 c2) h w -> b c1 c2 h w", c1=1, c2=3)
    
    b, c1, c2, h, w = pred_mask.shape 
    indicators = torch.tensor(indicators, dtype=pred_mask.dtype)
    indicators = repeat(indicators, "b d -> b c d h w", c=1, h=h, w=w)
    pred_mask = pred_mask * indicators
    gt_mask = gt_mask * indicators
    
    # Same as "compute_all_loss"
    return compute_all_loss(pred_mask, gt_mask, iou_pred)

def compute_all_loss(pred_mask, gt_mask, iou_pred):
    if len(pred_mask.shape) == 4:
        pred_mask = pred_mask.unsqueeze(1)
    if len(gt_mask.shape) == 4:
        gt_mask = gt_mask.unsqueeze(1)
    # import ipdb; ipdb.set_trace()
    fl, dl = combined_loss(pred_mask, gt_mask)
    segment_loss = 20*fl.mean() + dl.mean()
    iou_loss = mse_loss(iou_pred, compute_iou(torch.tensor(pred_mask>0, dtype=gt_mask.dtype), gt_mask))
    total_loss = segment_loss.mean() + iou_loss.mean()
    return total_loss, fl, dl, iou_loss


# def compute_


if __name__ == "__main__":
    pred_mask = torch.ones((1,9,1024,1024))*9
    pred_mask[:,:,:200,:] = -1
    gt_mask = torch.ones((1,3,1024,1024))
    loss = ranked_combined_loss(pred_mask, gt_mask, iou_pred=torch.ones(gt_mask.shape[:1]))
    print(loss)