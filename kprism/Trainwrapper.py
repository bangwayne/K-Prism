import logging
import numpy as np
import torch
from .KPrism_model import KPrism
from .utils.point_sampler import PointSampler
from .modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
import random

log = logging.getLogger()


class KPrismTrainWrapper(KPrism):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.iter_num = cfg.training.iter_num
        self.training_click_mode = cfg.training.training_click_mode
        self.sampling_probs = cfg.training.sampling_probs
        self.click_loss_weight = cfg.training.click_loss_weight
        self.num_ref = cfg.data.num_ref

    def forward(self, batched_inputs):
        losses = self.iter_training(batched_inputs)
        return losses

    def iter_training(self, batched_inputs):
        """
        Run one training iteration over a batch.

        Args:
            batched_inputs (list[dict]): each dict contains:
                - "image"  : Tensor (C, H, W)
                - "target" : dict with keys "labels" and "masks"
                - "q_index": int, query class index
                - "ref_img": Tensor (num_ref, 3, H, W), reference images
                - "ref_mask": Tensor (num_ref, 1, H, W), reference masks
                - "epoch"  : int, current epoch number

        Training modes (sampled with configured probabilities):
            '1' - Query-based segmentation with iterative point refinement
            '2' - Reference-guided segmentation with iterative point refinement
            '3' - Point-only segmentation (no initial query, pure click-based)
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = torch.stack(images, dim=0)
        height, width = images[0].shape[-2], images[0].shape[-1]

        sampled_element = random.choices(self.training_click_mode, weights=self.sampling_probs, k=1)[0]
        mode = [sampled_element]
        query_index = [x["q_index"] for x in batched_inputs]
        Sampler = PointSampler()
        batch_data_with_point = Sampler.initial_test_points(batch_data=batched_inputs, device=self.device)

        total_iter_num = self.iter_num

        # Mode 1: query-based segmentation with iterative point refinement
        if '1' in mode:
            features = self.backbone(images)
            for iter_num in range(total_iter_num):
                for data_dict in batch_data_with_point:
                    data_dict['click_index'] = iter_num
                if iter_num == 0:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_point).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                    pos_point_tuple_list, neg_point_tuple_list = None, None
                else:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                outputs = self.sem_seg_head(
                    features,
                    point_feature_batch_data,
                    point_tuple_list,
                    pos_point_tuple_list,
                    neg_point_tuple_list,
                    click_mode='1' if iter_num != 0 else '0',
                    query_index=query_index,
                )

                if "target" in batched_inputs[0]:
                    gt_instances = [x["target"] for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                if iter_num == 0:
                    losses = self.criterion(outputs, targets)
                else:
                    iter_losses = self.criterion(outputs, targets)
                    for key in iter_losses:
                        losses[key] += iter_losses[key] * self.click_loss_weight

                processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                current_batch = batch_data_with_point if iter_num == 0 else batch_data_with_next_point
                for bs_index in range(len(current_batch)):
                    current_batch[bs_index]['seg_result'] = processed_mask_results[bs_index]

                if iter_num < total_iter_num - 1:
                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point = Sampler.get_next_points(
                            current_batch, device=self.device, click_index=(iter_num + 1))
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point = Sampler.get_next_points_component(
                            current_batch, device=self.device, click_index=(iter_num + 1))

        # Mode 2: reference-guided segmentation with iterative point refinement
        if '2' in mode:
            object_summaries, pixel_readout = self.process_ref_inputs(batched_inputs, images)
            for iter_num in range(total_iter_num):
                for data_dict in batch_data_with_point:
                    data_dict['click_index'] = iter_num
                if iter_num == 0:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_point).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                    pos_point_tuple_list, neg_point_tuple_list = None, None
                else:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                outputs = self.sem_seg_head(
                    pixel_readout,
                    point_feature_batch_data,
                    point_tuple_list,
                    pos_point_tuple_list,
                    neg_point_tuple_list,
                    click_mode='4' if iter_num != 0 else '3',
                    query_index=query_index,
                    task_query=object_summaries
                )

                if "target" in batched_inputs[0]:
                    gt_instances = [x["target"] for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                if iter_num == 0:
                    losses = self.criterion(outputs, targets)
                else:
                    iter_losses = self.criterion(outputs, targets)
                    for key in iter_losses:
                        losses[key] += iter_losses[key] * self.click_loss_weight

                processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                current_batch = batch_data_with_point if iter_num == 0 else batch_data_with_next_point
                for bs_index in range(len(current_batch)):
                    current_batch[bs_index]['seg_result'] = processed_mask_results[bs_index]

                if iter_num < total_iter_num - 1:
                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point = Sampler.get_next_points(
                            current_batch, device=self.device, click_index=(iter_num + 1))
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point = Sampler.get_next_points_component(
                            current_batch, device=self.device, click_index=(iter_num + 1))

        # Mode 3: point-only segmentation (no initial query, pure click-based refinement)
        if '3' in mode:
            features = self.backbone(images)
            query_index = [x["q_index"] for x in batched_inputs]
            for iter_num in range(total_iter_num):
                for data_dict in batch_data_with_point:
                    data_dict['click_index'] = iter_num
                if iter_num == 0:
                    for bs_index in range(len(batch_data_with_point)):
                        batch_data_with_point[bs_index]['seg_result'] = torch.zeros(1, height, width)

                    batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                        batch_data_with_point, device=self.device, click_index=(iter_num + 1))

                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point_mode2).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point_mode2]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point_mode2]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point_mode2]

                    outputs = self.sem_seg_head(features,
                                                point_feature_batch_data,
                                                point_tuple_list,
                                                pos_point_tuple_list,
                                                neg_point_tuple_list,
                                                click_mode='2',
                                                query_index=query_index)

                    if "target" in batched_inputs[0]:
                        gt_instances = [x["target"] for x in batched_inputs]
                        targets = self.prepare_targets(gt_instances, images)
                    else:
                        targets = None

                    losses2 = self.criterion(outputs, targets)
                    if 'losses' not in locals():
                        losses = {key: torch.zeros_like(losses2[key]) for key in losses2}
                    for key in losses2:
                        losses[key] += losses2[key]
                    processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                    for bs_index in range(len(batch_data_with_next_point_mode2)):
                        batch_data_with_next_point_mode2[bs_index]['seg_result'] = processed_mask_results[bs_index]

                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point_mode2 = Sampler.get_next_points(
                            batch_data_with_next_point_mode2, device=self.device, click_index=(iter_num + 2))
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                            batch_data_with_next_point_mode2, device=self.device, click_index=(iter_num + 2))
                else:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point_mode2).to(self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point_mode2]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point_mode2]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point_mode2]
                    outputs = self.sem_seg_head(features,
                                                point_feature_batch_data,
                                                point_tuple_list,
                                                pos_point_tuple_list,
                                                neg_point_tuple_list,
                                                click_mode='2',
                                                query_index=query_index)

                    if "target" in batched_inputs[0]:
                        gt_instances = [x["target"] for x in batched_inputs]
                        targets = self.prepare_targets(gt_instances, images)
                    else:
                        targets = None

                    iter_losses = self.criterion(outputs, targets, cal_class_loss=False)
                    for key in iter_losses:
                        losses[key] += iter_losses[key] * self.click_loss_weight

                    processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                    for bs_index in range(len(batch_data_with_next_point_mode2)):
                        batch_data_with_next_point_mode2[bs_index]['seg_result'] = processed_mask_results[bs_index]

                    if iter_num < total_iter_num - 1:
                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point_mode2 = Sampler.get_next_points(
                                batch_data_with_next_point_mode2, device=self.device, click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                                batch_data_with_next_point_mode2, device=self.device, click_index=(iter_num + 2))

        return losses

    def process_ref_inputs(self, batched_inputs, images):
        """
        Extract support features from reference images and masks.

        Args:
            batched_inputs (list[dict]): each dict contains:
                - "ref_img"  : Tensor (num_ref, 3, H, W)
                - "ref_mask" : Tensor (num_ref, 1, H, W)
            images (Tensor): query image batch (B, 3, H, W)

        Returns:
            object_summaries (list[Tensor]): object-level summary features per scale
            pixel_readout    (dict[str, Tensor]): pixel-level support readout per scale key
        """
        ref_img = [x['ref_img'].to(self.device) for x in batched_inputs]
        ref_mask = [x['ref_mask'].to(self.device) for x in batched_inputs]
        ref_img = torch.stack(ref_img, dim=0)   # [B, R, 3, H, W]
        ref_mask = torch.stack(ref_mask, dim=0)  # [B, R, 1, H, W]

        b, all_num_ref, c, h, w = ref_img.shape
        k = 1
        indices = random.sample(range(all_num_ref), k)
        ref_img = ref_img[:, indices]
        ref_mask = ref_mask[:, indices]

        b, num_ref, _, h, w = ref_img.shape
        ref_img = ref_img.reshape(-1, 3, h, w)    # [B*R, 3, H, W]
        ref_mask = ref_mask.reshape(-1, 1, h, w)  # [B*R, 1, H, W]

        query_features = self.backbone(images)
        key_features = self.backbone(ref_img)
        ref_combine = torch.cat((ref_img, ref_mask), dim=1)  # [B*R, 4, H, W]
        value_features = self.mask_encoder(ref_combine)

        object_summaries = []
        pixel_readout = []
        ordered_keys = ["res5", "res4", "res3"]
        for key in ordered_keys:
            scale = self.feature_idx[key]
            query_feat = query_features[key]
            key_feat = key_features[key]
            value_feat = value_features[key]

            object_summary = self.object_summarizer(ref_mask, value_feat, scale, num_ref)
            object_summaries.append(object_summary)

            if key == "res5":
                key_feat = key_feat.reshape(b, num_ref, *key_feat.shape[1:]).contiguous()
                value_feat = value_feat.reshape(b, num_ref, *value_feat.shape[1:]).contiguous()
                align_data = self.read_support(query_feat, key_feat, value_feat)
                pixel_readout.append(align_data)
            else:
                pixel_readout.append(query_feat)

        pixel_readout = dict(zip(ordered_keys, pixel_readout))
        return object_summaries, pixel_readout
