import os
import sys
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import hydra
from omegaconf import OmegaConf, DictConfig
from kprism.inference.inference import InferenceCore
from kprism.data_mapper.test_data_utils import *
from utils.analysis import get_iou, get_dice
from utils.visualize_2d import draw_result_with_point
import pandas as pd
from tqdm import tqdm
import random


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


def gather_results(results, world_size):
    all_results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(all_results, results)
    merged_results = {}
    for res in all_results:
        merged_results.update(res)
    return merged_results


def muti_gpu_iter_inference(gpu, ngpus_per_node, cfg):
    node_rank = int(cfg.testing.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = ngpus_per_node * cfg.testing.num_nodes
    print(f"[Rank {rank}]: Use GPU: {gpu} for inference")
    is_main_host = rank == 0
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=cfg.testing.init_method,
        rank=rank,
        world_size=world_size,
    )

    model = InferenceCore(cfg).to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=cfg.testing.bucket_cap_md
    )

    if os.path.isfile(cfg.testing.resume):
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(cfg.testing.resume, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded checkpoint '{cfg.testing.resume}' (epoch {checkpoint['epoch']})")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f} M")
    else:
        raise FileNotFoundError(f"Checkpoint file '{cfg.testing.resume}' not found.")

    model.eval()
    test_dataloader = get_loader(cfg)
    test_iterator = tqdm(test_dataloader, desc=f"[RANK {rank}: GPU {gpu}]", dynamic_ncols=True)

    results = {}
    results_dice = {}
    for i, batch_data in enumerate(test_iterator):
        bs = len(batch_data)
        processed_results_dict, point_dict = model(batch_data)
        iter_num = len(processed_results_dict)
        for it in range(iter_num):
            for bs_index in range(bs):
                file_name = batch_data[bs_index]['file_name']
                slice_index = batch_data[bs_index]['slice_index']
                file_name = os.path.splitext(os.path.basename(file_name))[0] + "_" + str(slice_index)
                image = batch_data[bs_index]['ori_image']
                gt = batch_data[bs_index]['ori_sem_seg']

                result = processed_results_dict[it][bs_index]
                q_index = batch_data[bs_index]['unique_label']
                point = point_dict[it][bs_index]
                gt_mask = (gt == q_index).long()
                single_result = result[0].squeeze(0)

                if torch.max(gt_mask) > 0:
                    if cfg.testing.draw_plot:
                        work_dir = f"outputs/plots/{cfg.dataset.dataset_name}"
                        dice = get_dice(gt_mask, single_result)
                        draw_result_with_point(image, single_result, gt_mask, work_dir,
                                               catalog=q_index, slice_name=file_name,
                                               point_tuple=point, iter_num=it, dice=dice)

                    iou = get_iou(gt_mask, single_result)
                    dice = get_dice(gt_mask, single_result)
                    col_name = f"{file_name}_{q_index}"
                    if col_name not in results:
                        results[col_name] = [None] * 21
                        results_dice[col_name] = [None] * 21
                    results[col_name][it] = iou
                    results_dice[col_name][it] = dice

    gathered_results = gather_results(results, world_size)
    gathered_results_dice = gather_results(results_dice, world_size)

    if rank == 0:
        results_df = pd.DataFrame(gathered_results)
        results_dice_df = pd.DataFrame(gathered_results_dice)
        results_df.index.name = 'Iteration'
        results_dice_df.index.name = 'Iteration'

        dataset_name = cfg.dataset.dataset_name
        mode = cfg.testing.testing_click_mode[0]
        os.makedirs("outputs/results", exist_ok=True)
        iou_path = f"outputs/results/mode{mode}_{dataset_name}_iou.csv"
        dice_path = f"outputs/results/mode{mode}_{dataset_name}_dice.csv"
        results_df.to_csv(iou_path, index=True)
        results_dice_df.to_csv(dice_path, index=True)
        print(f"Results saved to {iou_path} and {dice_path}")
        print(results_dice_df.mean(axis=1))


@hydra.main(config_path="kprism/config",
            config_name="eval_config.yaml",
            version_base="1.3")
def main(cfg: DictConfig):
    set_seed(2025)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(cfg.testing.port)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.testing.gpus))
    ngpus_per_node = len(cfg.testing.gpus)
    cfg.testing.ngpus = ngpus_per_node

    print(f"Spawning {ngpus_per_node} processes for testing")
    mp.set_start_method('spawn', force=True)
    mp.spawn(muti_gpu_iter_inference, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))


if __name__ == "__main__":
    main()
    sys.exit(0)
