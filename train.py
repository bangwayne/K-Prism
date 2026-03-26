import os
import sys
import json
import torch
import argparse
import numpy as np
import shutil
import logging
from datetime import datetime, timedelta
import torch.multiprocessing as mp
import torch.distributed as dist
import hydra
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import time

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from kprism.Trainwrapper import KPrismTrainWrapper
from kprism.data_mapper.train_data_utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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


def setup_logger(name="train_logger", save_dir=None, filename=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            filename = f"train_{timestamp}.log"
        file_path = os.path.join(save_dir, filename)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    return logger


def muti_gpu_train_epoch(cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, rank, gpu,
                         train_iter_num):
    logger = logging.getLogger(f"rank{rank}_logger")
    epoch_loss = 0
    if rank == 0:
        epoch_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True)
    else:
        epoch_iterator = train_dataloader

    if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
        train_dataloader.sampler.set_epoch(epoch)

    model.train()

    for batch in epoch_iterator:
        for batch_data in batch:
            batch_data['epoch'] = epoch

        loss_dict = model(batch)
        if isinstance(loss_dict, dict):
            losses = sum(v for v in loss_dict.values())
        else:
            losses = loss_dict
        losses = losses.mean()

        if torch.isnan(losses).any():
            assert not torch.isnan(losses).any(), "NaN detected in loss. Training stopped."

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        if train_iter_num % 500 == 0 and rank == 0:
            logger.info(f"[EPOCH {epoch}] [ITER {train_iter_num}] Loss: {losses.item():.6f}")
        train_iter_num += 1
        epoch_loss += losses.item()

    scheduler.step()

    epoch_loss_tensor = torch.tensor(epoch_loss).to(gpu)
    dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
    epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()
    epoch_loss /= len(train_dataloader) + 1e-12

    if rank == 0:
        logger.info(f'[GPU {gpu}] epoch_loss: {epoch_loss}')
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr(), epoch)
    return epoch_loss, train_iter_num


def muti_gpu_main_worker(gpu, ngpus_per_node, cfg):
    node_rank = int(cfg.training.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node * cfg.training.num_nodes
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    torch.cuda.set_device(gpu)

    log_dir = os.path.join(cfg.training.model_save_path, "logs")
    logger = setup_logger(name=f"rank{rank}_logger", save_dir=log_dir, filename=f"train_rank{rank}.log")

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=cfg.training.init_method,
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
        bucket_cap_mb=cfg.training.bucket_cap_md
    )

    cfg_solver = cfg.solver
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_solver.base_lr,
        weight_decay=cfg_solver.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=cfg_solver.warmup_epochs,
        max_epochs=cfg.training.epochs,
        eta_min=cfg_solver.end_lr_ratio * cfg_solver.base_lr
    )

    num_epochs = cfg.training.epochs
    iter_num = 0
    train_dataloader = get_loader(cfg)
    start_epoch = 0

    if cfg.training.resume is not None:
        if os.path.isfile(cfg.training.resume):
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(cfg.training.resume, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            iter_num = checkpoint.get('iter_num', 0)
            scheduler.last_epoch = start_epoch
            logger.info(f"Loaded checkpoint from '{cfg.training.resume}' (epoch {start_epoch})")
        else:
            logger.warning(f"No checkpoint found at '{cfg.training.resume}'")

    if rank == 0:
        writer = SummaryWriter(log_dir='./tb_log/' + cfg.training.run_id)
    else:
        writer = None

    os.makedirs(cfg.training.model_save_path, exist_ok=True)
    for epoch in range(start_epoch, num_epochs):
        epoch_loss, iter_num = muti_gpu_train_epoch(
            cfg, model, train_dataloader, optimizer, scheduler, writer, epoch, rank, gpu, iter_num)

        logger.info(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

        if is_main_host and epoch == 2:
            cfg_save_path = os.path.join(cfg.training.model_save_path, 'config.json')
            with open(cfg_save_path, 'w') as cfg_file:
                json.dump(OmegaConf.to_container(cfg, resolve=True), cfg_file, indent=4)

        if is_main_host and epoch > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
                'iter_num': iter_num
            }
            torch.save(checkpoint, os.path.join(cfg.training.model_save_path, 'kprism_latest.pth'))

        if is_main_host and epoch % 5 == 0 and epoch > 40:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
                'iter_num': iter_num
            }
            torch.save(checkpoint, os.path.join(cfg.training.model_save_path, f'kprism_epoch_{epoch}.pth'))

        torch.distributed.barrier()


@hydra.main(config_path="kprism/config",
            config_name="train_config.yaml",
            version_base="1.3")
def main(cfg: DictConfig):
    set_seed(2025)

    cfg.training.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    cfg.training.model_save_path = os.path.join(cfg.training.work_dir, cfg.training.run_id)

    log_dir = os.path.join(cfg.training.model_save_path, "logs")
    logger = setup_logger(save_dir=log_dir)
    logger.info("Logger initialized.")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(cfg.training.port)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.training.gpus))
    ngpus_per_node = len(cfg.training.gpus)
    cfg.training.ngpus = ngpus_per_node

    print(f"Spawning {ngpus_per_node} processes for training")
    mp.set_start_method('spawn', force=True)
    mp.spawn(muti_gpu_main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))


if __name__ == "__main__":
    main()
    sys.exit(0)
