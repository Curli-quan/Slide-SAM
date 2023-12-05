import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from tutils.new.trainer.trainer_abstract import AbstractTrainer
from tutils.new.manager.loggers import MultiLogger
from tutils.new.manager.csv_recorder import CSVLogger
from tutils.new.trainer.recorder import Recorder
from tutils.new.utils.core_utils import _get_time_str
from tutils.new.utils.public_utils import dict_to_str

# Waiting for update
from tutils import tfilename, tenum

# export MASTER_ADDR=192.168.1.100
# export MASTER_PORT=12345


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def ddp(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def get_logger(config):
    config_base = config['base']
    config_logger = config['logger']
    logger = MultiLogger(logdir=config_base['runs_dir'], 
                        record_mode=config_logger.get('record_mode', None), 
                        tag=config_base['tag'], 
                        extag=config_base.get('experiment', None),
                        action=config_logger.get('action', 'k')) # backup config.yaml
    return logger

class DDPTrainer(AbstractTrainer):
    def __init__(self, config, tester=None, monitor=None, rank='cuda', world_size=0, logger=None):
        super().__init__(config, tester, monitor, rank, world_size)
        self.rank = rank
        self.logger = logger
        self.logging_available =  (self.rank == 0 or self.rank == 'cuda')
        print("Running on ", rank)
        self.global_iteration = 0
        if self.logging_available:
            print(f"Logger at Process(rank={rank})")
            self.recorder = Recorder(reduction=self.recorder_mode)
            self.recorder_valid = Recorder(reduction=self.recorder_mode)
            self.recorder_test = Recorder(reduction=self.recorder_mode)
            self.logger = None
            self.csvlogger = CSVLogger(tfilename(self.runs_dir, "best_record"))
            self.csvlogger_all = CSVLogger(tfilename(self.runs_dir, "all_record"))
            self.monitor = monitor
            self.tester = tester
            
            self.logger = get_logger(config)
            assert self.logger is not None, f"{__file__} Gotrank {self.rank}"
        
        if self.use_amp:
            self.scalar = GradScaler()
            print("Debug settings: use amp=",self.use_amp)

    def init_model(self, model, trainset, validset=None, **kwargs):
        # Use CacheDataset
        # trainset = CacheDataset(trainset, num_workers=12, cache_rate=0.5)
        if trainset is not None:
            assert len(trainset) > 0 , f"{__file__} Got{len(trainset)}"
            self.trainloader = DataLoader(dataset=trainset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=True,
                                        drop_last=True,
                                        pin_memory=True)
        if validset is not None:
            self.validloader = DataLoader(dataset=validset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        shuffle=True,
                                        drop_last=True,
                                        pin_memory=True)
        if self.load_pretrain_model:
            model.module.load()
        rank = self.rank
        model = model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        return ddp_model

    def configure_optim(self, model, **kwargs):
        # Set optimizer and scheduler
        optim_configs = model.module.configure_optimizers()
        assert isinstance(optim_configs, dict)
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        if self.load_optimizer:
            start_epoch = model.module.load_optim(optimizer)
            print(f"[DDPTrainer] Continue training, from epoch {start_epoch}")
        else:
            start_epoch = self.start_epoch
        return optimizer, scheduler, start_epoch
    
    def fit(self, model, trainset, validset=None):
        model = self.init_model(model, trainset, validset=validset, rank=self.rank)
        self.init_timers()
        optimizer, scheduler, start_epoch = self.configure_optim(model)

        for epoch in range(start_epoch, self.max_epochs):
            self.on_before_zero_grad()
            # Training
            self.timer_epoch()
            do_training_log = (epoch % self.training_log_interval == 0)
            if self.validloader is not None and self.validation_interval > 0 and epoch % self.validation_interval == 0:
                self.valid(model, self.validloader, epoch, do_training_log)
                
            if self.trainloader is not None:
                self.train(model, self.trainloader, epoch, optimizer, scheduler, do_training_log)

            if self.logging_available:
                self.test(model, epoch=epoch)

                if epoch % self.save_interval == 0 and self.logging_available:
                    if 'latest' in self.save_mode:
                        self.save(model, epoch, 'latest', optimizer)
                    if 'all' in self.save_mode:
                        self.save(model, epoch, None, optimizer)
                    # time_save_model = self.timer_epoch()

        print("Training is Over for GPU rank ", self.rank)
        self.cleanup()

    def test(self, model, epoch):
        # Evaluation
        if epoch % self.val_check_interval == 0 and self.logging_available:
            print("Note: Tester runs on <rank 0> only")
            if self.tester is not None:
                out = self.tester.test(model=model, epoch=epoch, rank=self.rank)
                if self.monitor is not None:
                    best_dict = self.monitor.record(out, epoch)
                    self.recorder_test.record({**best_dict, **out})
                    if best_dict['isbest']:
                        if 'best' in self.save_mode:
                            self.save(model, epoch, type='best')
                        self.csvlogger.record({**best_dict, **out, "time": _get_time_str()})
                    if self.save_all_records:
                        self.csvlogger_all.record({**best_dict, **out, "time": _get_time_str()})
                    self.logger.info(f"\n[*] {dict_to_str(best_dict)}[*] Epoch {epoch}: \n{dict_to_str(out)}")
                    self.logger.add_scalars(out, step=epoch, tag='test')
                    # if ''
                else:
                    self.logger.info(f"\n[*] Epoch {epoch}: {dict_to_str(out)}")
                self.on_after_testing(d=out)

    def save(self, model, epoch, type=None, optimizer=None, **kwargs):
        if self.logging_available:
            if type is None:
                # if self.save_interval > 0 and epoch % self.save_interval == 0:
                save_name = "/ckpt/model_epoch_{}.pth".format(epoch)
                model.module.save(tfilename(self.runs_dir, save_name), epoch=epoch)
                self.logger.info(f"Epoch {epoch}: Save model to ``{save_name}``! ")
            elif type == 'best':
                # save_name = "/ckpt/best_model_epoch_{}.pth".format(epoch)
                save_name2 = "/ckpt_v/model_best.pth"
                # model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_best=True)
                model.module.save(tfilename(self.runs_dir, save_name2), epoch=epoch, is_best=True)
                self.logger.info(f"[Best model] Epoch {epoch}: Save model to ``{save_name2}``! ")
            elif type == 'latest':
                if self.save_interval > 0 and epoch % self.save_interval == 0:
                    save_name = "/ckpt_v/model_latest.pth"
                    model.module.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_latest=True)
                    save_optim_name = "/ckpt/optim_latest.pth"
                    model.module.save_optim(tfilename(self.runs_dir, save_optim_name), optimizer=optimizer, epoch=epoch)
                    self.logger.info(f"Epoch {epoch}: Save checkpoint to ``{save_name}``")
            elif type == "iteration":
                save_name = "/ckpt/model_iter_{}.pth".format(self.global_iteration)
                model.module.save(tfilename(self.runs_dir, save_name), epoch=self.global_iteration)
                self.logger.info(f"Epoch {epoch}: Save model to ``{save_name}``! ")


    def train(self, model, trainloader, epoch, optimizer, scheduler=None, do_training_log=True):
        model.train()
        out = {}
        if do_training_log and self.logging_available:
            self.recorder.clear()
            time_record = 0.1111
            self.timer_batch()

        success_count = 0
        failed_count = 0
        for load_time, batch_idx, data in tenum(trainloader):
            optimizer.zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            if self.use_amp:           
                with autocast():
                    self.timer_net()
                    out = model.module.training_step(data, batch_idx, epoch=epoch)
                    # try:
                    #     out = model.module.training_step(data, batch_idx, epoch=epoch)
                    # except Exception as e:
                    #     msg = f"Ignore Error! {e}"
                    #     if self.logging_available:
                    #         self.logger.info(msg)
                    #     else:
                    #         print(msg)
                    #     continue
                    assert isinstance(out, dict)
                    time_fd = self.timer_net()
                    loss = out['loss']
                    self.scalar.scale(loss).backward()
                    self.scalar.step(optimizer)
                    self.scalar.update()
                    time_bp = self.timer_net()
            else:
                self.timer_net()
                try:
                    out = model.module.training_step(data, batch_idx, epoch=epoch)
                except Exception as e:
                    msg = f"Ignore Error! {e}"
                    if self.logging_available:
                        self.logger.info(msg)
                    else:
                        print(msg)
                    continue
                if out['loss'] is None:
                    failed_count += 1
                    continue
                if torch.isnan(out['loss']):
                    print("Ignore Nan Value: ", out['loss'])
                    failed_count += 1
                    # raise ValueError(f"Get loss: {out['loss']}")
                assert isinstance(out, dict)
                time_fd = self.timer_net()
                loss = out['loss']
                loss.backward()
                optimizer.step()
                time_bp = self.timer_net()
                success_count += 1

            time_batch = self.timer_batch()
            # batch logger !
            if self.logging_available and do_training_log:
                out['time_load'] = load_time
                out['time_cuda'] = time_data_cuda
                out['time_forward'] = time_fd
                out['time_bp'] = time_bp
                out['time_record'] = time_record
                out['time_batch'] = time_batch
                self.timer_data()
                self.recorder.record(out)
                time_record = self.timer_data()
                
                # for debug !
                if epoch == 0:
                    if self.logging_available:
                        self.logger.info("[*] Debug Checking Pipeline !!!")
                    del out 
                    return
                if self.global_iteration % 100 == 0:
                    print(f"Epoch: {epoch} | batch:{batch_idx}/{len(trainloader)}, Iteration:{self.global_iteration}, results: {to_item(out)}", end='\n')
                if self.global_iteration % 5000 == 0:
                    self.save(model, epoch, "iteration", optimizer)
                    # print("")
            self.global_iteration += 1

        if scheduler is not None:
            scheduler.step()        
    
        # epoch logger !
        if self.logging_available:
            if do_training_log :
                _dict = self.recorder.cal_metrics()
                _dict['time_total'] = self.timer_epoch()

                # print(_dict)
                # assert isinstance(lr, float), f"Got lr={lr}, type: {type(lr)}"
                loss_str = ""
                for k, v in _dict.items():
                    loss_str += "{}:{:.4f} ".format(k, v)
                # lr = optimizer.param_groups[0]['lr']
                lr = self.get_lr(optimizer)
                _dict['lr'] = lr
                loss_str += "{}:{:.6e} ".format('lr', lr)
                self.logger.info(f"Epoch {epoch}: {loss_str}")
                # _dict_with_train_tag = {f"train/{k}":v for k,v in _dict.items()}
                self.logger.add_scalars(_dict, step=epoch, tag='train')
                time_log_scalars = self.timer_epoch()
                self.on_after_training(d=_dict)
        # Clear
        del out
        del data
 
    def valid(self, model, validloader, epoch, do_training_log=True):
        model.eval()
        out = {}
        if do_training_log and self.logging_available:
            self.recorder_valid.clear()
            time_record = 0.1
            self.timer_batch()
        
        success_count = 1
        failed_count = 1
        for load_time, batch_idx, data in tenum(validloader):
            # model.on_before_zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            if self.use_amp:
                with autocast():
                    self.timer_net()
                    out = model.module.validation_step(data, batch_idx, epoch=epoch)
                    assert isinstance(out, dict)
                    time_fd = self.timer_net()
            else:
                self.timer_net()
                out = model.module.validation_step(data, batch_idx, epoch=epoch)
                if out['loss'] is None:
                    failed_count += 1
                    continue
                if torch.isnan(out['loss']):
                    self.logger.info("Nan Value: ", out['loss'])
                    failed_count += 1
                    raise ValueError(f"Get loss: {out['loss']}")
                assert isinstance(out, dict)
                time_fd = self.timer_net()
                success_count += 1

            time_batch = self.timer_batch()
            # batch logger !
            if do_training_log and self.logging_available:
                out['time_load'] = load_time
                out['time_cuda'] = time_data_cuda
                out['time_forward'] = time_fd
                out['time_record'] = time_record
                out['time_batch'] = time_batch
                self.timer_data()
                self.recorder_valid.record(out)
                time_record = self.timer_data()
            
            if batch_idx % 2 == 0:
                print(f"Valid Epoch: {epoch}. Processing batch_idx:{batch_idx} / {len(validloader)}, time_load: {load_time}, results: {to_item(out)}", end='\r')
                
            if epoch == 0:
                if self.logging_available:
                    self.logger.info("[*] Debug Checking validation Pipeline !!!")
                del out 
                return
            # model.on_after_zero_grad(d=out)
        if self.logging_available:
            self.logger.info(f"Training Success Ratio: {success_count / (success_count + failed_count)}")

        # epoch logger !
        if self.logging_available:
            if do_training_log :
                _dict = self.recorder_valid.cal_metrics()
                _dict['time_total'] = self.timer_epoch()

                # print(_dict)
                # assert isinstance(lr, float), f"Got lr={lr}, type: {type(lr)}"
                loss_str = ""
                for k, v in _dict.items():
                    loss_str += "{}:{:.4f} ".format(k, v)
                self.logger.info(f"Epoch {epoch}: {loss_str}")
                # _dict_with_val_tag = {f"val/{k}":v for k,v in _dict.items()}
                self.logger.add_scalars(_dict, step=epoch, tag='val')
                time_log_scalars = self.timer_epoch()
                self.on_after_training(d=_dict)
        # Clear
        del out
        del data 

        

def to_item(tensors):
    for k,v in tensors.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.detach().cpu().item()
    return tensors