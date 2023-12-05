"""
    Volume evalutaion

"""
import torch
import numpy as np
from torch.utils.data import DataLoader
# from datasets.dataset3d import Dataset3D
from tutils.new.manager import ConfigManager
from datasets.eval_dataloader.loader_abstract import AbstractLoader

from core.volume_predictor import VolumePredictor
from datasets.data_engine import DataManager, BoxPromptGenerator, PointPromptGenerator

from tutils import tfilename
from tutils.new.trainer.recorder import Recorder
from trans_utils.metrics import compute_dice_np
from trans_utils.data_utils import Data3dSolver



class Evaluater:
    def __init__(self, config) -> None:
        self.config = config
        self.recorder = Recorder()

    def solve(self, model, dataset):
        # model.eval()
        self.predictor = model
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for i, data in enumerate(dataloader):
            # if i <4: 
            #     print
            #     continue
            # for k, v in data.items():
            #     if isinstance(v, torch.Tensor):
            #         data[k] = v.to(self.rank)  
            if self.config['dataset']['prompt'] == 'box':
                res = self.eval_step(data, batch_idx=i)  
            if self.config['dataset']['prompt'] == 'point':
                res = self.eval_step_point(data, batch_idx=i)  
            self.recorder.record(res)
        res = self.recorder.cal_metrics()
        print(res)
        print("prompt:", self.config['dataset']['prompt'], " class_idx:", self.config['dataset']['label_idx'])

    def eval_step(self, data, batch_idx=0):
        name = data['name']
        dataset_name = data['dataset_name'][0]
        label_idx = data['label_idx'][0]
        template_slice_id = data['template_slice_id'][0]

        assert data['img'].shape[1] >= 3, f" Got img.shape {data['img'].shape}"
        if template_slice_id == 0:
            template_slice_id += 1
        elif template_slice_id == (data['img'].shape[0] - 1):
            template_slice_id -= 1

        spacing = data['spacing'].numpy().tolist()[0]
        if data['img'].shape[-1] < 260:
            # assert data['img'].shape[-1] < 260, f"Got {data['img'].shape}"
            img = data['img'][0][:,:256,:256]
            label = data['label'][0][:,:256,:256]
        else:
            img = data['img'][0]
            label = data['label'][0]
        # img = torch.clip(img, -200, 600)
        box = BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id].detach().cpu().numpy())
        box = np.array([box])
        pred, stability = self.predictor.predict_volume(
            x=img,
            box=box,            
            template_slice_id=template_slice_id,
            return_stability=True,
        )
        prompt_type = 'box'
        dice = compute_dice_np(pred, label.detach().cpu().numpy())
        Data3dSolver().simple_write(pred, path=tfilename(f"visual/{dataset_name}/pred_{batch_idx}_label_{label_idx}_{prompt_type}.nii.gz"), spacing=spacing)
        Data3dSolver().simple_write(label.detach().cpu().numpy(), path=tfilename(f"visual/{dataset_name}/label_{batch_idx}.nii.gz"))
        # Data3dSolver().simple_write(img.detach().cpu().numpy(), path=tfilename(f"visual/{dataset_name}/img_{batch_idx}.nii.gz"))
        # np.save(tfilename(f"meta/{dataset_name}/stability_{batch_idx}.npy"), stability)
        print(dataset_name, name, dice)
        return {"dice": dice}
    
    def eval_step_point(self, data, batch_idx=0):
        name = data['name']
        dataset_name = data['dataset_name'][0]
        label_idx = data['label_idx'][0]
        template_slice_id = data['template_slice_id'][0]
        spacing = data['spacing'].numpy().tolist()[0]

        assert data['img'].shape[1] >= 3, f" Got img.shape {data['img'].shape}"
        if template_slice_id == 0:
            template_slice_id += 1
        elif template_slice_id == (data['img'].shape[0] - 1):
            template_slice_id -= 1

        if data['img'].shape[-1] < 260:
            # assert data['img'].shape[-1] < 260, f"Got {data['img'].shape}"
            img = data['img'][0][:,:256,:256]
            label = data['label'][0][:,:256,:256]
        else:
            img = data['img'][0]
            label = data['label'][0]
       
        box = BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id].detach().cpu().numpy())
        point = (box[0]+box[2])*0.5 , (box[1]+box[3])*0.5
        point = np.array([point]).astype(int)
        if label[template_slice_id][point[0,1], point[0,0]] == 0:
            print("Use random point instead !!!")
            point = PointPromptGenerator().get_prompt_point(label[template_slice_id])
            point = np.array([point]).astype(int)
        # box = np.array([box])
        pred = self.predictor.predict_volume(
            x=img,
            point_coords=point,  
            point_labels=np.ones_like(point)[:,:1],
            template_slice_id=template_slice_id,   
        )
        dice = compute_dice_np(pred, label.detach().cpu().numpy())
        prompt_type = 'point'
        Data3dSolver().simple_write(pred, path=tfilename(f"visual/{dataset_name}/pred_{batch_idx}_label_{label_idx}_{prompt_type}.nii.gz"), spacing=spacing)
        # Data3dSolver().simple_write(pred, path=tfilename(f"visual/{dataset_name}/pred_{batch_idx}.nii.gz"))
        print(dataset_name, name, dice)
        return {"dice": dice}

def to_RGB(img):
    pass

if __name__ == "__main__":
    # from core.learner3 import SamLearner
    # from modeling.build_sam3d import sam_model_registry

    from core.learner3 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry

    EX_CONFIG = {       
        'dataset':{
            'prompt': 'box',
            'dataset_list': ['word'], # ["sabs"], chaos, word
            'label_idx': 1,
        }       
    }

    config = ConfigManager()
    config.add_config("configs/vit_b_103.yaml")
    config.add_config(EX_CONFIG)
    
    # Init Model
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=None)
    learner = SamLearner(sam_model=sam, config=config, data_engine=DataManager(img_size=(1024,1024)))
    learner.use_lora()
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b3/lora+edge2/ckpt_v/model_latest.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b3/lora+edge2/ckpt/model_epoch_20.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b3/lora+edge2/ckpt/model_epoch_16.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b3/lora_small/ckpt/model_epoch_6.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b9/lora/ckpt/model_epoch_50.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b11/spec_8/ckpt_v/model_latest.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b11/spec_5/ckpt/model_epoch_100.pth"
    pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b9/lora3/ckpt/model_iter_360000.pth"
    # pth = "/home1/quanquan/code/projects/finetune_large/runs/sam/ddp_b9/lora3/ckpt/model_iter_500000.pth"
    learner.load_well_trained_model(pth)
    learner.cuda()
    predictor = VolumePredictor(
        model=learner.model, 
        use_postprocess=True,
        use_noise_remove=True,)

    solver = Evaluater(config)
    dataset = AbstractLoader(config['dataset'], split="test")
    solver.solve(predictor, dataset)