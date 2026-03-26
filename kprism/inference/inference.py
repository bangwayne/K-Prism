import logging
import numpy as np
import torch
import random
from torch.nn import functional as F
from .resize_transform import SegmentationPreprocessor
from ..KPrism_model import KPrism
from ..utils.point_sampler import PointSampler
from ..modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map

log = logging.getLogger()


class InferenceCore(KPrism):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.iter_num = cfg.testing.iter_num
        self.testing_click_mode = cfg.testing.testing_click_mode
        self.num_ref = cfg.testing.num_ref
        self.long_size = cfg.model.setting.long_side_size
        self.size_preprocessor = SegmentationPreprocessor(long_size=self.long_size)

    def forward(self, batched_inputs):
        result_dict, point_dict = self.iter_inference(batched_inputs)
        return result_dict, point_dict

    def iter_inference(self, batched_inputs, unseen_class=False):
        """
        Iterative point-based inference.

        Args:
            batched_inputs (list[dict]): each dict contains:
                - "image"     : Tensor (C, H, W)
                - "q_index"   : int, query class index
                - "ref_img"   : Tensor (num_ref, 3, H, W)  [required for mode '2']
                - "ref_mask"  : Tensor (num_ref, 1, H, W)  [required for mode '2']
                - "size_info" : (H, W) original image size before padding
                - "pad_info"  : padding applied during preprocessing
                - "scale_factor": scale applied during preprocessing

        Returns:
            result_dict (dict[int, list[Tensor]]): predicted masks per iteration
            point_dict  (dict[int, list]):         point prompts used per iteration
        """
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = torch.stack(images, dim=0)
            height, width = images[0].shape[-2], images[0].shape[-1]
            mode = self.testing_click_mode

            # Mode 1: query-based segmentation with iterative point refinement
            if '1' in mode:
                query_index = [int(x["q_index"]) for x in batched_inputs]
                point_sampler = PointSampler()
                batch_data_with_point = point_sampler.initial_test_points(batch_data=batched_inputs, device=self.device)
                result_dict = {}
                point_dict = {}
                features = self.backbone(images)
                for iter_num in range(self.iter_num):
                    if iter_num == 0:
                        for batch_dict in batch_data_with_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data=batch_data_with_point).to(self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                        outputs = self.sem_seg_head(features,
                                                    point_feature_batch_data,
                                                    point_tuple_list,
                                                    click_mode='0',
                                                    query_index=query_index)

                        processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                        for bs_index in range(len(batch_data_with_point)):
                            batch_data_with_point[bs_index]['seg_result'] = processed_mask_results[bs_index]

                        seg_result_list, point_list = self._restore_outputs(batch_data_with_point)
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(
                                batch_data_with_point, device=self.device, click_index=(iter_num + 1))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_point, device=self.device, click_index=(iter_num + 1))

                    else:
                        for batch_dict in batch_data_with_next_point:
                            batch_dict['click_index'] = iter_num
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
                            click_mode='1',
                            query_index=query_index,
                        )

                        processed_results = F.interpolate(outputs["pred_masks"], size=(height, width),
                                                          mode="bilinear", align_corners=False).sigmoid()
                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_results[bs_index]

                        seg_result_list, point_list = self._restore_outputs(batch_data_with_next_point)
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 1))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 1))

            # Mode 2: reference-guided segmentation with iterative point refinement
            if '2' in mode:
                features = self.backbone(images)
                query_index = [int(x["q_index"]) for x in batched_inputs]
                object_summaries, pixel_readout = self.process_ref_inputs(batched_inputs, images, mode="test")
                point_sampler = PointSampler()
                batch_data_with_point = point_sampler.initial_test_points(batch_data=batched_inputs, device=self.device)
                result_dict = {}
                point_dict = {}
                for iter_num in range(self.iter_num):
                    if iter_num == 0:
                        for batch_dict in batch_data_with_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data=batch_data_with_point).to(self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                        outputs = self.sem_seg_head(pixel_readout,
                                                    point_feature_batch_data,
                                                    point_tuple_list,
                                                    click_mode='3',
                                                    query_index=query_index,
                                                    task_query=object_summaries)
                        processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                        for bs_index in range(len(batch_data_with_point)):
                            batch_data_with_point[bs_index]['seg_result'] = processed_mask_results[bs_index]

                        seg_result_list, point_list = self._restore_outputs(batch_data_with_point)
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(
                                batch_data_with_point, device=self.device, click_index=(iter_num + 1))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_point, device=self.device, click_index=(iter_num + 1))

                    else:
                        for batch_dict in batch_data_with_next_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                        pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                        neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                        outputs = self.sem_seg_head(
                            pixel_readout, point_feature_batch_data, point_tuple_list,
                            pos_point_tuple_list, neg_point_tuple_list,
                            click_mode='4',
                            query_index=query_index,
                            task_query=object_summaries
                        )
                        processed_results = F.interpolate(outputs["pred_masks"], size=(height, width),
                                                          mode="bilinear", align_corners=False).sigmoid()
                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_results[bs_index]

                        seg_result_list, point_list = self._restore_outputs(batch_data_with_next_point)
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 1))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 1))

            # Mode 3: point-only segmentation (no initial query)
            if '3' in mode:
                features = self.backbone(images)
                query_index = [int(x["q_index"]) for x in batched_inputs]
                point_sampler = PointSampler()
                batch_data_with_point = point_sampler.initial_test_points(batch_data=batched_inputs, device=self.device)
                result_dict = {}
                point_dict = {}
                for iter_num in range(self.iter_num):
                    if iter_num == 0:
                        for batch_dict in batch_data_with_point:
                            batch_dict['click_index'] = iter_num
                        for bs_index in range(len(batch_data_with_point)):
                            batch_data_with_point[bs_index]['seg_result'] = torch.zeros(1, height, width)

                        batch_data_with_next_point = point_sampler.get_next_points_component(
                            batch_data_with_point, device=self.device, click_index=(iter_num + 1))
                        point_feature_batch_data = get_batch_point_feature_map(
                            batch_data=batch_data_with_next_point).to(self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                        pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                        neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                        outputs = self.sem_seg_head(features,
                                                    point_feature_batch_data,
                                                    point_tuple_list,
                                                    pos_point_tuple_list,
                                                    neg_point_tuple_list,
                                                    click_mode='2',
                                                    query_index=query_index)

                        processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_mask_results[bs_index]

                        seg_result_list, point_list = self._restore_outputs(batch_data_with_point)
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 2))

                    else:
                        point_list = [x['points_list'] for x in batch_data_with_next_point]
                        for batch_dict in batch_data_with_next_point:
                            batch_dict['click_index'] = iter_num
                        point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(self.device)
                        point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                        pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                        neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                        outputs = self.sem_seg_head(features,
                                                    point_feature_batch_data,
                                                    point_tuple_list,
                                                    pos_point_tuple_list,
                                                    neg_point_tuple_list,
                                                    click_mode='2',
                                                    query_index=query_index)

                        processed_results = F.interpolate(outputs["pred_masks"], size=(height, width),
                                                          mode="bilinear", align_corners=False).sigmoid()
                        for bs_index in range(len(batch_data_with_next_point)):
                            batch_data_with_next_point[bs_index]['seg_result'] = processed_results[bs_index]

                        seg_result_list, point_list = self._restore_outputs(batch_data_with_point)
                        result_dict[iter_num] = seg_result_list
                        point_dict[iter_num] = point_list

                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point = point_sampler.get_next_points(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point = point_sampler.get_next_points_component(
                                batch_data_with_next_point, device=self.device, click_index=(iter_num + 2))

            return result_dict, point_dict

    def _restore_outputs(self, batch_data):
        """Unpad and rescale segmentation masks and point coordinates to original image space."""
        seg_result_list = []
        point_list = []
        for item in batch_data:
            seg_result = item['seg_result'].detach().cpu()
            size_info = item['size_info']
            pad_info = item['pad_info']
            scale_factor = item['scale_factor']
            restored_seg = self.size_preprocessor.unpad_and_resize(
                seg_result, original_size=size_info, pad=pad_info)
            restore_point = self.size_preprocessor.map_valid_points_back(
                item['points_list'][0][0],
                item['points_list'][0][1],
                original_size=size_info,
                pad=pad_info,
                scale_factor=scale_factor,
            )
            seg_result_list.append(restored_seg)
            point_list.append(restore_point)
        return seg_result_list, point_list

    def process_ref_inputs(self, batched_inputs, images, mode="train"):
        """
        Extract support features from reference images and masks.

        Args:
            batched_inputs (list[dict]): each dict contains 'ref_img' (num_ref, 3, H, W)
                                         and 'ref_mask' (num_ref, 1, H, W).
            images (Tensor): query images (B, 3, H, W)
            mode (str): "train" randomly samples references; "test" uses all references.

        Returns:
            object_summaries (list[Tensor]): object-level summary features per scale
            pixel_readout    (dict[str, Tensor]): pixel-level readout per scale key
        """
        ref_img = [x['ref_img'].to(self.device) for x in batched_inputs]
        ref_mask = [x['ref_mask'].to(self.device) for x in batched_inputs]
        ref_img = torch.stack(ref_img, dim=0)
        ref_mask = torch.stack(ref_mask, dim=0)

        b, all_num_ref, c, h, w = ref_img.shape
        if mode == "train":
            k = random.choice([1, 2, 3])
            indices = random.sample(range(all_num_ref), k)
            ref_img = ref_img[:, indices]
            ref_mask = ref_mask[:, indices]

        b, num_ref, _, h, w = ref_img.shape
        ref_img = ref_img.reshape(-1, 3, h, w)
        ref_mask = ref_mask.reshape(-1, 1, h, w)

        query_features = self.backbone(images)
        key_features = self.backbone(ref_img)
        ref_combine = torch.cat((ref_img, ref_mask), dim=1)
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
