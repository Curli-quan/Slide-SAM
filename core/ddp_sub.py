"""
    from ddp_b9.py
    
    Add additional bypass/side-way to finetune on other datasets
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tutils import tfilename, tdir

from datasets.dataset3d_2dmask import Dataset2D
# from datasets.dataset3d import Dataset3D
from datasets.cache_dataset3d3 import Dataset3D
from datasets.dataset_merged import DatasetMerged, TestsetMerged
from datasets.data_engine import DataEngine
from modeling.build_sam3d2 import sam_model_registry

from .learner_sub1 import SamLearner
# from tutils.new.trainer.trainer_ddp import DDPTrainer
from trans_utils.trainer_ddp import DDPTrainer
# from .lora_sam import LoRA_Sam

import warnings
warnings.filterwarnings("ignore")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def ddp_train(rank, world_size, config):    
    setup(rank, world_size)
    
    # sam_checkpoint = "/quanquan/code/segment-anything/segment_anything/sam_vit_b_01ec64.pth" # A800 server
    # sam_checkpoint = "/home1/quanquan/code/projects/medical-guangdong/segment-anything/sam_vit_b_01ec64.pth" # 103 server
    model_type = "vit_b"
    device = rank 

    config_data = config['dataset']
    data_type = config_data.get("types", ["3d", "2d"])
    data_type = [data_type] if isinstance(data_type, str) else data_type
    dataset = Dataset3D(config_data, split='train')

    # assert len(validset) > 0
    data_engine = DataEngine(dataset=dataset, img_size=(1024,1024))
    sam = sam_model_registry[model_type](checkpoint=None)

    learner = SamLearner(sam_model=sam, config=config, data_engine=data_engine)
    learner.use_lora()
    learner.load_well_trained_model(config['training']['breakpoint_path']) # use preset path
    learner.use_lora_sub()

    ddp_trainer = DDPTrainer(config=config, rank=rank, world_size=world_size)
    ddp_trainer.fit(learner, trainset=data_engine, validset=None)

    cleanup()


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def run_demo(demo_fn, world_size, config):
    mp.spawn(demo_fn,
             args=(world_size,config),
             nprocs=world_size,
             join=True)
    
from collections import OrderedDict
import yaml
import yamlloader
def _ordereddict_to_dict(d):
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if isinstance(v, OrderedDict):
            v = _ordereddict_to_dict(v)
            d[k] = dict(v)
        elif type(v) == list:
            d[k] = _ordereddict_to_dict(v)
        elif isinstance(v, dict):
            d[k] = _ordereddict_to_dict(v)
    return d
    
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m core.ddp_b3 --tag lora --config configs/vit_b_103.yaml

if __name__ == "__main__":
    import argparse
    from tutils.new.manager import trans_args, trans_init, ConfigManager

    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but {__file__} Got{n_gpus}"
    if n_gpus == 1:
        print("Warning! Running on only 1 GPU! just for debug")
    world_size = n_gpus
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/vit_sub.yaml")
    parser.add_argument("--func", default="train")
    parser.add_argument("--reuse", action="store_true")

    args = trans_args(parser=parser)
    config = ConfigManager()
    config.auto_init(file=__file__, args=args, ex_config=None)
    # config.save()
    path = tfilename(config['base']['runs_dir'], "config.yaml")
    with open(path, "w") as f:
        yaml.dump(_ordereddict_to_dict(config), f)
        print("Save config file to ", path)

    if n_gpus < 1: exit(0)
    run_demo(ddp_train, world_size, config)
