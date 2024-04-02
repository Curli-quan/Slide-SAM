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
from tutils.tutils.ttimer import timer


class Evaluater:
    def __init__(self, config) -> None:
        self.config = config
        self.recorder = Recorder()

    def solve(self, model, dataset, finetune_number=1):
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
            # try:
            if True:
                if self.config['dataset']['prompt'] == 'box':
                    dice, pred, label, temp_slice = self.eval_step(data, batch_idx=i)  
                    used_slice = [temp_slice]
                    if finetune_number > 1:
                        for i in range(finetune_number - 1):    
                            dice, pred, label, temp_slice = self.finetune_with_more_prompt(pred, label, exclude_slide_id=used_slice)
                            used_slice.append(temp_slice)
                    res = {"dice": dice}
                if self.config['dataset']['prompt'] == 'point':
                    res = self.eval_step_point(data, batch_idx=i)  
                self.recorder.record(res)
            # except Exception as e:
            #     print(e)
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
        print("Using slice ", template_slice_id, " as template slice")

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
        # Data3dSolver().simple_write(pred, path=tfilename(f"visual/{dataset_name}/pred_{batch_idx}_label_{label_idx}_{prompt_type}.nii.gz"), spacing=spacing)
        # Data3dSolver().simple_write(label.detach().cpu().numpy(), path=tfilename(f"visual/{dataset_name}/label_{batch_idx}.nii.gz"))
        # Data3dSolver().simple_write(img.detach().cpu().numpy(), path=tfilename(f"visual/{dataset_name}/img_{batch_idx}.nii.gz"))
        # np.save(tfilename(f"meta/{dataset_name}/stability_{batch_idx}.npy"), stability)
        print(dataset_name, name, dice)
        template_slice_id = template_slice_id if isinstance(template_slice_id, int) else template_slice_id.item()
        return dice, pred, label.detach().cpu().numpy(), template_slice_id
    
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

    def finetune_with_more_prompt(self, pred, label, prompt_type="box", exclude_slide_id=[]):
        assert pred.shape == label.shape
        dices = [compute_dice_np(pred[j,:,:], label[j,:,:]) for j in range(pred.shape[0])]
        rank_list = np.array(dices[1:-1]).argsort() # Ignore the head and tail
        rank_list += 1 # Ignore the head and tail
        for i in rank_list:
            if i in exclude_slide_id:
                continue
            template_slice_id = i
            break
        # template_slice_id += 1 # Ignore the head and tail
        print("Using slice ", template_slice_id, " as template slice")
        old_confidence = self.predictor.get_confidence()
        box = BoxPromptGenerator(size=None).mask_to_bbox(label[template_slice_id])
        box = np.array([box])
        new_pred, stability = self.predictor.predict_with_prompt(
            box=box,            
            template_slice_id=template_slice_id,
            return_stability=True,
        )
        new_confidence = self.predictor.get_confidence()
        new_confidence[template_slice_id] *= 2
        all_conf = np.stack([old_confidence, new_confidence], axis=1)
        preds = [pred, new_pred]
        merged = np.zeros_like(label)
        for slice_idx in range(pred.shape[0]):            
            idx = np.argsort(all_conf[slice_idx,:])[-1]
            merged[slice_idx,:,:] = preds[idx][slice_idx]

        print("old dices", [compute_dice_np(pred, label) for pred in preds])
        dice = compute_dice_np(merged, label)
        print("merged dice, idx", dice)
        return dice, merged, label, template_slice_id


def to_RGB(img):
    pass

if __name__ == "__main__":
    from core.learner3 import SamLearner
    from modeling.build_sam3d2 import sam_model_registry
    EX_CONFIG = {       
        'dataset':{
            'prompt': 'box',
            'prompt_number': 5,
            'dataset_list': ['example'], # ["sabs"], chaos, word pancreas
            'label_idx': 5,
            },
        "lora_r": 24,
        'model_type': "vit_h",
        'ckpt': "/home1/quanquan/code/projects/finetune_large/segment_anything/model_iter_3935000.pth",
    }
    config = ConfigManager()
    config.add_config("configs/vit_b.yaml")
    config.add_config(EX_CONFIG)
    config.print()
    dataset = AbstractLoader(config['dataset'], split="test")
    print(len(dataset))
    assert len(dataset) >= 1
    
    # Init Model
    model_type = config['model_type'] # "vit_b"
    sam = sam_model_registry[model_type](checkpoint=None)
    learner = SamLearner(sam_model=sam, config=config, data_engine=DataManager(img_size=(1024,1024)))
    learner.use_lora(r=config['lora_r'])
    pth = config['ckpt']
    learner.load_well_trained_model(pth)
    learner.cuda()
    predictor = VolumePredictor(
        model=learner.model, 
        use_postprocess=True,
        use_noise_remove=True,)

    solver = Evaluater(config)
    dataset = AbstractLoader(config['dataset'], split="test")
    solver.solve(predictor, dataset, finetune_number=config['dataset']['prompt_number'])