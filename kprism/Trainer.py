import os
import sys
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import torch.multiprocessing as mp
import shutil
import logging

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .Trainwrapper import KPrismTrainWrapper
import random
import time
import torch.distributed as dist


class Trainer:
    def __init__(self, model_cfg, train_cfg, log, run_path):
        self.use_amp = train_cfg.amp

        local_rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self.model = nn.parallel.DistributedDataParallel(
            KPrismTrainWrapper(model_cfg).cuda(),
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
            bucket_cap_mb=train_cfg.training.bucket_cap_md
        )

        self.train()
        parameter_groups = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=train_cfg.solver.base_lr,
            weight_decay=train_cfg.solver.weight_decay
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=train_cfg.solver.warmup_epochs,
            max_epochs=train_cfg.training.epochs
        )
        self.train_dataloader = get_loader(train_cfg)

        self.log = log
        self.run_path = run_path
        self.log.log_string('model_size',
                            str(sum([param.nelement() for param in self.model.parameters()])))
        self.log.log_string(
            'number_of_parameters_that_requires_gradient',
            str(sum([param.nelement() for param in
                     filter(lambda p: p.requires_grad, self.model.parameters())])))
        self.log.log_string('torch version', torch.__version__)

    def train(self):
        self._is_train = True
        self.model.train()
        return self

    def train_epoch(self, cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, gpu,
                    train_iter_num):
        rank = self.local_rank
        epoch_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True)
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        model.train()

        for batch in epoch_iterator:
            for batch_data in batch:
                batch_data['epoch'] = epoch

            loss_dict = model(batch)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
            else:
                losses = sum(loss_dict.values())

            if torch.isnan(losses).any():
                assert not torch.isnan(losses).any(), "NaN detected in loss. Training stopped."

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            print(f'[RANK {rank}: GPU {gpu}] EPOCH-{epoch} ITER-{train_iter_num} --- loss {losses.item()}')
            train_iter_num += 1
            if rank == 0:
                writer.add_scalar('train_iter/loss', losses, train_iter_num)
            epoch_loss += losses.item()

        scheduler.step()

        epoch_loss_tensor = torch.tensor(epoch_loss).to(gpu)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()
        epoch_loss /= len(train_dataloader) + 1e-12

        if rank == 0:
            writer.add_scalar('train/loss', epoch_loss, epoch)
            writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
        return epoch_loss, train_iter_num

    def save_checkpoint(self, it, epoch, save_copy=False):
        if self.local_rank != 0:
            return
        checkpoint = {
            'it': it,
            'epoch': epoch,
            'weights': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        model_save_name = "kprism_model_epoch_latest.pth"
        model_path = os.path.join(self.run_path, model_save_name)
        torch.save(checkpoint, model_path)
        self.log.info(f'Latest checkpoint saved to {model_path}.')

        if save_copy:
            model_save_name = f"kprism_model_epoch_{epoch}.pth"
            model_path = os.path.join(self.run_path, model_save_name)
            torch.save(checkpoint, model_path)
            self.log.info(f'Epoch {epoch} checkpoint saved to {model_path}.')

    def load_checkpoint(self, path):
        print(f"=> loading checkpoint '{path}'")
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})
        it = checkpoint['it']
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.log.info('Network weights, optimizer states, and scheduler states loaded.')
        return it, epoch


def set_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--config_file", type=str, default='')
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument('-work_dir', type=str, default='./work_dir')
    parser.add_argument('-num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    try:
        torch.distributed.destroy_process_group()
    except:
        pass
    sys.exit(0)


def muti_gpu_train_epoch(cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, rank, gpu, train_iter_num):
    epoch_loss = 0
    epoch_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True)
    if cfg.TRAINING.DIST:
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
    model.train()

    for batch in epoch_iterator:
        for batch_data in batch:
            batch_data['epoch'] = epoch

        loss_dict = model(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
        else:
            losses = sum(loss_dict.values())

        if torch.isnan(losses).any():
            assert not torch.isnan(losses).any(), "NaN detected in loss. Training stopped."

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        print(f'[RANK {rank}: GPU {gpu}] EPOCH-{epoch} ITER-{train_iter_num} --- loss {losses.item()}')
        train_iter_num += 1
        if rank == 0:
            writer.add_scalar('train_iter/loss', losses, train_iter_num)
        epoch_loss += losses.item()

    scheduler.step()

    if cfg.TRAINING.DIST:
        epoch_loss_tensor = torch.tensor(epoch_loss).to(gpu)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()

    epoch_loss /= len(train_dataloader) + 1e-12

    if rank == 0:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
    return epoch_loss, train_iter_num


def muti_gpu_main_worker(gpu, ngpus_per_node, cfg, args):
    node_rank = int(cfg.TRAINING.NODE_RANK)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node * cfg.TRAINING.NUM_NODES
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=cfg.TRAINING.INIT_METHOD,
        rank=rank,
        world_size=world_size,
    )

    model = KPrismTrainWrapper(cfg).to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=cfg.TRAINING.BUCKET_CAP_MD
    )

    if os.path.isfile(args.resume):
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=True)
        print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cfg_solver = cfg.SOLVER
    if cfg_solver.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_solver.BASE_LR,
            weight_decay=cfg_solver.WEIGHT_DECAY
        )

    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=cfg_solver.WARMUP_EPOCH,
                                              max_epochs=150)
    num_epochs = cfg.TRAINING.NUM_EPOCH
    iter_num = 0
    train_dataloader = get_loader(cfg)
    start_epoch = 0

    if cfg.TRAINING.RESUME is not None:
        if os.path.isfile(cfg.TRAINING.RESUME):
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(cfg.TRAINING.RESUME, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAINING.RESUME, checkpoint['epoch']))

    if rank == 0:
        writer = SummaryWriter(log_dir='./tb_log/' + cfg.TRAINING.RUN_ID)
        print('Writing Tensorboard logs to ', './tb_log/' + cfg.TRAINING.RUN_ID)
    else:
        writer = None

    for epoch in range(start_epoch, num_epochs):
        with model.join():
            epoch_loss, iter_num = muti_gpu_train_epoch(
                cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, rank, gpu, iter_num)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

        if is_main_host and epoch > 50:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(cfg.TRAINING.MODEL_SAVE_PATH, 'kprism_model_latest.pth'))

        torch.distributed.barrier()


def main(cfg):
    set_seed(2024)
    torch.cuda.empty_cache()
    if cfg.TRAINING.DIST:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        cfg.TRAINING.RUN_ID = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(cfg.TRAINING.WORK_DIR, cfg.TRAINING.RUN_ID)
        cfg.TRAINING.MODEL_SAVE_PATH = model_save_path
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9999'
        ngpus_per_node = 4
        specific_gpus = [0, 1, 2, 3]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, specific_gpus))
        cfg.TRAINING.NGPUS = ngpus_per_node
        print(f"Spawning processes, ngpus_per_node={ngpus_per_node}")
        print(f"=====> project save at {cfg.TRAINING.MODEL_SAVE_PATH}")
        mp.set_start_method('spawn')
        try:
            mp.spawn(muti_gpu_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, args))
        except KeyboardInterrupt:
            print("\n[MAIN] Training interrupted. Cleaning up...")
            cleanup()
    else:
        gpu = 0
        cfg.TRAINING.RUN_ID = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = os.path.join(cfg.TRAINING.WORK_DIR, cfg.TRAINING.RUN_ID)
        cfg.TRAINING.MODEL_SAVE_PATH = model_save_path
        print(f"=====> project save at {cfg.TRAINING.MODEL_SAVE_PATH}")
        os.makedirs(cfg.TRAINING.MODEL_SAVE_PATH, exist_ok=True)
        main_worker(gpu, cfg)


if __name__ == "__main__":
    args = set_parse()
    cfg = setup(args)
    main(cfg=cfg)
    sys.exit(0)
