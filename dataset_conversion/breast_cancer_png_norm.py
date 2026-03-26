import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random
import yaml
import imageio
from PIL import Image

import os
import shutil

def organize_images_masks_rename(source_path, target_path):
    image_dir = os.path.join(target_path, "image")
    mask_dir = os.path.join(target_path, "annotations")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for file in os.listdir(source_path):
        if not file.endswith(".png"):
            continue

        src = os.path.join(source_path, file)

        if file.endswith("_mask.png"):
            # Change mask name: belign (1)_mask.png → belign (1)_gtr.png
            new_name = file.replace("_mask.png", "_gt.png")
            dst = os.path.join(mask_dir, new_name)
        else:
            dst = os.path.join(image_dir, file)

        shutil.copyfile(src, dst)

    print(f"✅ Images copied to: {image_dir}")
    print(f"✅ Masks copied & renamed to: {mask_dir}")

def create_yaml(source_path, train_ratio=0.8, seed=42, save_dir="."):
    all_files = os.listdir(source_path)

    # Extract base names like "belign (1)" without "_mask"
    patient_names = set()
    for f in all_files:
        if f.endswith(".png") and not f.endswith("_mask.png"):
            base = f.replace(".png", "")
            patient_names.add(base)

    patient_names = sorted(list(patient_names))
    random.seed(seed)
    random.shuffle(patient_names)

    split_idx = int(len(patient_names) * train_ratio)
    train_list = patient_names[:split_idx]
    test_list = patient_names[split_idx:]
    os.makedirs(save_dir,exist_ok=True)
    train_yaml_path = os.path.join(save_dir, "train.yaml")
    test_yaml_path = os.path.join(save_dir, "test.yaml")

    with open(train_yaml_path, 'w') as f:
        yaml.dump({'patients': train_list}, f)
    with open(test_yaml_path, 'w') as f:
        yaml.dump({'patients': test_list}, f)

    print(f"✅ Saved train list ({len(train_list)} samples) → {train_yaml_path}")
    print(f"✅ Saved test list ({len(test_list)} samples) → {test_yaml_path}")


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
        img_name = f"{name}.png"
        label_name = f"{name}_mask.png"

        src_img = os.path.join(source_dir, img_name)
        src_label = os.path.join(mask_path, label_name)

        tgt_img = os.path.join(target_image_dir, img_name)
        tgt_label = os.path.join(target_label_dir, f"{name}_gt.png")

        # # 拷贝图像文件
        # if os.path.exists(src_img):
        #     shutil.copy(src_img, tgt_img)
        #     print(f"Copied image: {img_name}")
        # else:
        #     print(f"Image not found: {src_img}")
        #
        # # 拷贝标注文件
        # if os.path.exists(src_label):
        #     shutil.copy(src_label, tgt_label)
        #     print(f"Copied label: {label_name}")
        # else:
        #     print(f"Label not found: {src_label}")

        if os.path.exists(src_img) and os.path.exists(src_label):
            shutil.copy(src_img, tgt_img)
            print(f"Copied image: {img_name}")
            shutil.copy(src_label, tgt_label)
            print(f"Copied label: {label_name}")


# import imageio
# from PIL import Image

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
        if not name.endswith(".png"):
            continue

        base_name = name.replace(".png", "")
        img_path = os.path.join(img_dir, name)
        lab_path = os.path.join(lab_dir, f"{base_name}_gt.png")
        if not os.path.exists(lab_path):
            print(f"Missing label for {base_name}, skipping...")
            continue

        # Load 2D image and label
        img = np.array(Image.open(img_path)).astype(np.float32)
        lab = np.array(Image.open(lab_path)).astype(np.uint8) * 255
        # print(f"unique_label: {np.unique(lab)}")
        # print(lab.shape)
        # Normalize image to 0-255
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
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

# source_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/pool_data/benign"
# target_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/list"
# create_yaml(source_path, train_ratio=0.8, seed=42, save_dir=target_path)
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/list/train.yaml"
# source_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/pool_data/benign"
# target_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/rescale_data/train"
# copy_yaml_file(yaml_path, source_path, source_path, target_path)
# organize_images_masks_rename(source_path, target_path)
# source_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/CXR_png"
# create_yaml(source_path, train_ratio=0.8, seed=42, save_dir="/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/list")
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/list/test.yaml"
# source_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/CXR_png"
# mask_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/masks"
# target_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/rescale_data/test"
#
# copy_yaml_file(yaml_path, source_path, mask_path, target_path)

dataset_info = ('chest_xray', 'mri')
source_path = "/research/cbim/medical/bg654/verse_dataset/Breast_cancer/rescale_data"
target_path = "/research/cbim/medical/bg654/verse_dataset/Breast_cancer/Data/"
process_existing_2d_data(dataset_info, source_path, target_path, file="train")
process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# process_3d_data(dataset_info, source_path, target_path, file="train")
# process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'
