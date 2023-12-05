"""
    Use mask_decoder3d_2.py
"""

import torch
import torchvision
import numpy as np
from tutils.trainer import Trainer, LearnerModule
from einops import rearrange, repeat, reduce
import torch.optim.lr_scheduler as lr_scheduler

from core.loss import ranked_combined_loss_with_indicators
from .learner3 import SamLearner as basic_learner
from .loss import compute_all_loss, ranked_combined_loss, compute_iou, combined_loss


class SamLearner(basic_learner):
    
    def load_pretrained_model(self, pth, *args, **kwargs):        
        """
            Unmatched: prompt_encoder.mask_downscaling.0.weight
                their: torch.Size([4, 1, 2, 2])
                our: torch.Size([4, 3, 2, 2])
            Unmatched: mask_decoder.mask_tokens.weight
                their: torch.Size([4, 256])
                our: torch.Size([12, 256])
        """
        print("Load pretrained model for mask_decoder3d_2 !!")

        state_dict = torch.load(pth)
        model_state_dict = self.model.state_dict()
        model_state_dict.update(state_dict)
        model_state_dict['prompt_encoder.mask_downscaling.0.weight'] = repeat(state_dict['prompt_encoder.mask_downscaling.0.weight'], "a 1 c d -> a b c d", b=3)
        # model_state_dict['mask_decoder.mask_tokens.weight'] = repeat(state_dict['mask_decoder.mask_tokens.weight'], "a d -> (a 3) d")

        for k, v in model_state_dict.items():
            if k.startswith("mask_decoder.output_upscaling2"):
                k2 = k.replace("output_upscaling2.", "output_upscaling." )
                model_state_dict[k] = model_state_dict[k2]
                print("Load weights: ", k)
            if k.startswith("mask_decoder.output_upscaling3"):
                k2 = k.replace("output_upscaling3.", "output_upscaling." )
                model_state_dict[k] = model_state_dict[k2]
                print("Load weights: ", k)

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