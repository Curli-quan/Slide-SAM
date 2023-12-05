import numpy as np



def compute_dice_np(pred_mask, gt_mask):
    """ numpy values 
    """
    assert gt_mask.max() == 1, f"Got gt_mask.max():{gt_mask.max()} Error!!"
    pred_mask = np.array(pred_mask>0)
    gt_mask = np.array(gt_mask>0)
    intersection = np.array(pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    dice = intersection * 2 / union # if union > 0 else 0
    return dice


def compute_prec_np(pred_mask, gt_mask):
    true_pos = (np.int32(pred_mask>0) * np.int32(gt_mask>0)).sum()
    return true_pos / np.int32(pred_mask>0).sum()

def compute_recall_np(pred_mask, gt_mask):
    true_pos = (np.int32(pred_mask>0) * np.int32(gt_mask>0)).sum()
    false_neg = ((gt_mask - pred_mask)>0).sum()
    return true_pos / (true_pos + false_neg)    