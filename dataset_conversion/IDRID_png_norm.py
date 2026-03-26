import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random
import yaml
import imageio
from PIL import Image
import cv2
import os
import numpy as np
from glob import glob


# 路径设置
def copy_yaml_file(yaml_path, source_dir, mask_path, target_path):
    # yaml_path = "config.yaml"
    # source_dir = "/path/to/source"  # 原始数据存放路径
    os.makedirs(target_path, exist_ok=True)
    target_image_dir = os.path.join(target_path, "image")
    target_label_dir = os.path.join(target_path, "annotations")

    # 确保输出文件夹存在
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # 读取 YAML 文件
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # print(cfg)
    patient_list = cfg['patients']

    for name in patient_list:
        img_name = f"{name}.jpg"
        label_name = f"{name}.png"

        src_img = os.path.join(source_dir, img_name)
        src_label = os.path.join(mask_path, label_name)

        tgt_img = os.path.join(target_image_dir, f"{name}.png")
        tgt_label = os.path.join(target_label_dir, f"{name}_gt.png")

        if os.path.exists(src_img) and os.path.exists(src_label):
            shutil.copy(src_img, tgt_img)
            print(f"Copied image: {img_name}")
            shutil.copy(src_label, tgt_label)
            print(f"Copied label: {label_name}")


def normalize_red_green(label_rgb):
    """
    清除图像中蓝色通道后，对红色/绿色区域标准化颜色：
    - 红色区域：R=255, G=0, B=0
    - 绿色区域：R=0, G=255, B=0
    """
    label_rgb = label_rgb.copy()

    # 清除蓝色通道
    label_rgb[:, :, 2] = 0

    # 通道二值化
    label_rgb[label_rgb > 128] = 255
    label_rgb[label_rgb <= 128] = 0

    # 红绿区域掩膜
    red_mask = (label_rgb[:, :, 0] > 100) & (label_rgb[:, :, 0] >= label_rgb[:, :, 1])
    green_mask = (label_rgb[:, :, 1] > 100) & (label_rgb[:, :, 1] > label_rgb[:, :, 0])

    label_rgb[red_mask] = [255, 0, 0]
    label_rgb[green_mask] = [0, 255, 0]

    return label_rgb

def convert_rgb_mask_to_label(mask_rgb):
    """
    将 RGB 掩膜图转换为单通道标签图：
    - [255, 0, 0] → 1
    - [0, 255, 0] → 2
    - 其他        → 0
    """
    mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

    red_mask = np.all(mask_rgb == [255, 0, 0], axis=-1)
    green_mask = np.all(mask_rgb == [0, 255, 0], axis=-1)

    mask[red_mask] = 100
    mask[green_mask] = 200

    return mask

def process_existing_2d_data(dataset_info, source_path, target_path, file="train"):
    dataset, modality = dataset_info
    file_path = os.path.join(source_path, file)

    img_dir = os.path.join(file_path, "image")
    lab_dir = os.path.join(file_path, "annotations")

    save_img_dir = os.path.join(target_path, file, "image")
    save_lab_dir = os.path.join(target_path, file, "annotations")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lab_dir, exist_ok=True)

    for name in os.listdir(img_dir):

        base_name = name.replace(".jpg", "")
        img_path = os.path.join(img_dir, name)
        lab_path = os.path.join(lab_dir, f"{base_name}_OD.tif")
        if not os.path.exists(lab_path):
            print(f"Missing label for {base_name}, skipping...")
            continue

        # Load 2D image and label
        img = np.array(Image.open(img_path)).astype(np.float32)
        lab = np.array(Image.open(lab_path)).astype(np.uint8)
        print(f"unique_label: {np.unique(lab)}")
        print(lab.shape)
        # Normalize image to 0-255
        # img_min, img_max = img.min(), img.max()
        # if img_max > img_min:
        #     img = (img - img_min) / (img_max - img_min) * 255.0
        # else:
        #     img = np.zeros_like(img)
        img = np.clip(img,0,255)
        img = img.astype(np.uint8)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # (H, W) -> (H, W, 3)
        # Optionally skip blank labels
        if np.max(lab) == 0:
            print(f"Skipping {base_name}: empty label.")
            continue
        # # Save processed slices
        imageio.imwrite(os.path.join(save_img_dir, f"{base_name}.png"), img)
        imageio.imwrite(os.path.join(save_lab_dir, f"{base_name}_gt.png"), lab)

        print(f"✅ Processed {base_name}")


# 示例调用
# txt_folder = "/research/cbim/medical/bg654/verse_dataset/PAPILA/PapilaDB-PAPILA/ExpertsSegmentations/Contours"
# output_folder = "/research/cbim/medical/bg654/verse_dataset/PAPILA/PapilaDB-PAPILA/Mask"
# img_path = "/research/cbim/medical/bg654/verse_dataset/PAPILA/PapilaDB-PAPILA/FundusImages"
# process_disc_contours(txt_folder, img_path, output_folder)


# source_path =  "/research/cbim/medical/bg654/verse_dataset/PAPILA/PapilaDB-PAPILA/FundusImages"
# save_dir =  "/research/cbim/medical/bg654/verse_dataset/PAPILA/list"
# create_yaml(source_path, train_ratio=0.8, seed=42, save_dir=save_dir)

# yaml_path = "/research/cbim/medical/bg654/verse_dataset/PAPILA/list/test.yaml"
# source_dir = "/research/cbim/medical/bg654/verse_dataset/PAPILA/PapilaDB-PAPILA/FundusImages"
# mask_path = "/research/cbim/medical/bg654/verse_dataset/PAPILA/PapilaDB-PAPILA/Mask"
# target_path = "/research/cbim/medical/bg654/verse_dataset/PAPILA/rescale_data/test"
# copy_yaml_file(yaml_path, source_dir, mask_path, target_path)
# source_path = "/research/cbim/medical/bg654/verse_dataset/bkai-igh-neopolyp/train/train"
# create_yaml(source_path, train_ratio=0.8, seed=42, save_dir="/research/cbim/medical/bg654/verse_dataset/bkai-igh-neopolyp/list")
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/bkai-igh-neopolyp/list/test.yaml"
# source_path = "/research/cbim/medical/bg654/verse_dataset/bkai-igh-neopolyp/train/train"
# mask_path = "/research/cbim/medical/bg654/verse_dataset/bkai-igh-neopolyp/train_gt/train_gt"
# target_path = "/research/cbim/medical/bg654/verse_dataset/bkai-igh-neopolyp/rescale_data/test"
# copy_yaml_file(yaml_path, source_path, mask_path, target_path)

dataset_info = ('chest_xray', 'mri')
source_path = "/research/cbim/medical/bg654/verse_dataset/IDRiD/rescale_data"
target_path = "/research/cbim/medical/bg654/verse_dataset/IDRiD/Data/"
process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# process_3d_data(dataset_info, source_path, target_path, file="train")
# process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'
