from typing import Tuple
import torch
import random
from torch import nn
from torch.nn import functional as F
from .modeling.loss import PointSampleLoss
from .utils.point_sampler import PointSampler
from .modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
from .utils.memory_utils import *
from .modeling.backbone.utnet import UTNet
from .modeling.meta_arch.seghead import SegHead
from .modeling.task_encoder.mask_encoder import MaskEncoder
from .modeling.task_encoder.object_summarier import ObjectSummarizer


class KPrism(nn.Module):
    """
    Main model class for KPrism: a point-prompted, reference-guided medical image segmentation framework.

    Components:
        backbone       : UTNet for multi-scale feature extraction
        sem_seg_head   : SegHead (PixelFuser + transformer decoder) for mask prediction
        mask_encoder   : encodes reference image+mask pairs
        object_summarizer : aggregates object-level features from reference support
        criterion      : PointSampleLoss (dice + sigmoid cross-entropy)
    """

    def __init__(self, cfg, *, deep_supervision=True):
        super().__init__()
        model_cfg = cfg.model
        self.backbone = UTNet(model_cfg)
        self.sem_seg_head = SegHead(model_cfg, self.backbone.output_shape())
        self.criterion = PointSampleLoss()
        self.object_summarizer = ObjectSummarizer(model_cfg)
        self.mask_encoder = MaskEncoder(model_cfg)
        self.num_queries = model_cfg.model.task_encoder.num_queries
        self.size_divisibility = model_cfg.setting.size_divisibility
        self.feature_idx = model_cfg.model.object_summarizer.feature_idx

        pixel_mean = model_cfg.setting.pixel_mean
        pixel_std = model_cfg.setting.pixel_std
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.point_sample_method = model_cfg.setting.point_sample_method

        dice_weight = model_cfg.setting.dice_weight
        mask_weight = model_cfg.setting.mask_weight
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = model_cfg.model.transformer_decoder.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

    @property
    def device(self):
        return self.pixel_mean.device

    def single_inference(self, batched_inputs, outputs):
        """Upsample predicted masks to input image resolution."""
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = torch.stack(images, dim=0)

            mask_pred_results = outputs["pred_masks"]
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        return mask_pred_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image['masks'].to(self.device)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image['labels'].to(self.device),
                    "masks": padded_masks,
                }
            )
        return new_targets

    @staticmethod
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        sem_seg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return sem_seg

    @staticmethod
    def read_support(query_key: torch.Tensor,
                     memory_key: torch.Tensor,
                     msk_value: torch.Tensor) -> torch.Tensor:
        """
        Attention-based readout of reference features.

        query_key   : B * CK * H * W
        memory_key  : B * CK * T * H * W
        msk_value   : B * CV * H * W
        returns     : B * CV * H * W  pixel-level readout
        """
        affinity = get_affinity(memory_key.float(), query_key.float())
        msk_value = msk_value.float()
        pixel_readout = readout(affinity, msk_value)
        return pixel_readout

    @staticmethod
    def read_sim_rank(query_key: torch.Tensor,
                      memory_key: torch.Tensor) -> torch.Tensor:
        """
        query_key  : B * CK * H * W
        memory_key : B * CK * T * H * W
        returns    : best-match index per query position
        """
        memory_key = memory_key.transpose(1, 2)
        _, best_match_idx = get_similarity(memory_key.float(), query_key.float(), rank=True)
        return best_match_idx

    def forward(self, *args, **kwargs):
        raise NotImplementedError
