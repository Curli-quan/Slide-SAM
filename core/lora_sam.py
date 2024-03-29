# Modified from https://github.com/JamesQFreeman/Sam_LoRA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
# from modeling.sam3d import Sam


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.in_features = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv

class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
        freeze_all: freeze whole sam, otherwise only image encoder (VIT)

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model, r: int, lora_layer:[int]=None, freeze_all:bool=False, freeze_prompt_encoder=True):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        if freeze_all:
            for param in sam_model.parameters():
                param.requires_grad = False
        else:
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = False
            for param in sam_model.image_encoder.patch_embed.parameters():
                param.requires_grad = True
        if freeze_prompt_encoder:
            for param in sam_model.prompt_encoder.parameters():
                param.requires_grad = False            
                
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model
        # with open('vit_named_para.txt', 'w') as f:
        #     for k, v in sam_model.image_encoder.named_parameters():
        #         f.write(f'{k} {v.shape}\n')


    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """
        # assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
            # import ipdb; ipdb.set_trace()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=5**0.5)
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    from segment_anything import sam_model_registry
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator  # prompt and every mode
    import numpy as np
    lora_r = 8
    path = "../checkpoints/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=None)
    print('before lora', get_parameter_number(sam))
    lora_sam = LoRA_Sam(sam, lora_r)
    print('after lora', get_parameter_number(sam))
    x = torch.rand(size=(3,1024,1024))
    path = '../data/cache/data2d_3/0007_s0069_img.npy'
    img = np.load(path)
    print('img shape', img.shape)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(x)
    print('mask num', len(masks))
    # loss = np.sum([mask.mean(-1).mean(-1) for mask in masks])
    # predictor = SamPredictor(sam)
    # predictor.set_image(x)
    # masks, _, _ = predictor.predict([50,50])

    for f_name in ['save_lora_parameters', 'load_lora_parameters']:
        print(f_name)
        getattr(lora_sam, f_name)('tmp.safetensors')
