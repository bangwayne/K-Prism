from datetime import timedelta
from pathlib import Path
import torch
import numpy as np


def get_iou(gt_mask, pred_mask, ignore_label=-1):

    pred_mask = (pred_mask > 0.5).int()

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    iou = intersection / union if union != 0 else 1.0
    return iou


def get_dice(gt_mask, pred_mask, ignore_label=-1):
    # Apply a threshold to the predicted mask
    pred_mask = (pred_mask > 0.5).int()

    # Convert torch tensors to numpy arrays if necessary
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    # Generate masks to ignore the ignored labels and identify object of interest
    valid_mask = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    # Calculate intersection and union for the Dice coefficient
    intersection = np.logical_and(pred_mask, obj_gt_mask).sum()
    union = pred_mask.sum() + obj_gt_mask.sum()

    # Calculate Dice coefficient
    dice = (2. * intersection) / union if union != 0 else 1.0

    return dice


def get_dice_array(gt_mask, pred_mask, ignore_label=-1):
    # Apply a threshold to the predicted mask
    pred_mask = (pred_mask > 0.5).astype(int)

    # Generate masks to ignore the ignored labels and identify object of interest
    valid_mask = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    # Calculate intersection and union for the Dice coefficient
    intersection = np.logical_and(pred_mask, obj_gt_mask).sum()
    union = pred_mask.sum() + obj_gt_mask.sum()

    # Calculate Dice coefficient
    dice = (2. * intersection) / union if union != 0 else 1.0

    return dice


# Multi-label evaluation functions
def get_iou_multilabel(gt_mask, pred_mask, ignore_label=-1):
    """
    Calculate IoU for multi-label segmentation.
    Computes per-class IoU and returns the mean.
    
    Args:
        gt_mask: Ground truth mask with class indices (0, 1, 2, 3, ...)
        pred_mask: Prediction mask with class indices (0, 1, 2, 3, ...)
        ignore_label: Label to ignore (default: -1)
    
    Returns:
        Mean IoU across all classes
    """
    # Convert to numpy if needed
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    
    # Ensure pred_mask is integer class indices
    if pred_mask.max() <= 1.0 and pred_mask.min() >= 0.0:
        # If pred_mask is still in [0,1] range, convert to class indices
        pred_mask = (pred_mask > 0.5).astype(np.int32)
    else:
        pred_mask = pred_mask.astype(np.int32)
    
    gt_mask = gt_mask.astype(np.int32)
    
    # Get unique classes (excluding background 0 and ignore_label)
    unique_classes = np.unique(gt_mask)
    unique_classes = unique_classes[(unique_classes != 0) & (unique_classes != ignore_label)]
    
    if len(unique_classes) == 0:
        return 1.0  # No valid classes
    
    ious = []
    for cls in unique_classes:
        gt_cls = (gt_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = np.logical_or(gt_cls, pred_cls).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0)  # Both empty, perfect match
    
    return np.mean(ious) if len(ious) > 0 else 1.0


def get_dice_multilabel(gt_mask, pred_mask, ignore_label=-1):
    """
    Calculate Dice coefficient for multi-label segmentation.
    Computes per-class Dice and returns the mean.
    
    Args:
        gt_mask: Ground truth mask with class indices (0, 1, 2, 3, ...)
        pred_mask: Prediction mask with class indices (0, 1, 2, 3, ...)
        ignore_label: Label to ignore (default: -1)
    
    Returns:
        Mean Dice coefficient across all classes
    """
    # Convert to numpy if needed
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    
    # Ensure pred_mask is integer class indices
    if pred_mask.max() <= 1.0 and pred_mask.min() >= 0.0:
        # If pred_mask is still in [0,1] range, convert to class indices
        pred_mask = (pred_mask > 0.5).astype(np.int32)
    else:
        pred_mask = pred_mask.astype(np.int32)
    
    gt_mask = gt_mask.astype(np.int32)
    
    # Get unique classes (excluding background 0 and ignore_label)
    unique_classes = np.unique(gt_mask)
    unique_classes = unique_classes[(unique_classes != 0) & (unique_classes != ignore_label)]
    
    if len(unique_classes) == 0:
        return 1.0  # No valid classes
    
    dices = []
    for cls in unique_classes:
        gt_cls = (gt_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(gt_cls, pred_cls).sum()
        union = gt_cls.sum() + pred_cls.sum()
        if union > 0:
            dices.append((2. * intersection) / union)
        else:
            dices.append(1.0)  # Both empty, perfect match
    
    return np.mean(dices) if len(dices) > 0 else 1.0