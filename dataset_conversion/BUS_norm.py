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

def rename_masks_to_gt(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            old_path = os.path.join(folder_path, file)
            new_name = file.replace(".png", "_gt.png")
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file} → {new_name}")

# 用法
# rename_masks_to_gt("/common/users/bg654/verse_dataset/BUS/test/annotations")

# 路径设置
# def copy_file(source_folder, target_path, surname="."):
#     # yaml_path = "config.yaml"
#     # source_dir = "/path/to/source"  # 原始数据存放路径
#     os.makedirs(target_path, exist_ok=True)
#     target_image_dir = os.path.join(target_path, "image")
#     target_label_dir = os.path.join(target_path, "annotations")

#     # 确保输出文件夹存在
#     os.makedirs(target_image_dir, exist_ok=True)
#     os.makedirs(target_label_dir, exist_ok=True)

#     # 读取 YAML 文件
#     # print(cfg)
#     for filename in os.listdir(source_folder):
#         if filename.endswith("_orig.jpg") or filename.endswith("_contour.png"):
#             patient_name = filename.split('_')[0]

#             source_path = os.path.join(source_folder, filename)

#             # Copy based on file type
#             if filename.endswith("_orig.jpg"):
#                 dest_path = os.path.join(target_image_dir, f"{patient_name + surname}.png")
#             else:  # _contour.png
#                 dest_path = os.path.join(target_label_dir, f"{patient_name + surname}_gt.png")

#             shutil.copy2(source_path, dest_path)
#             print(f"Copied {filename} → {dest_path}")


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
        # if not name.endswith(".jpg"):
        #     continue

        base_name = name.replace(".png", "")
        img_path = os.path.join(img_dir, name)
        lab_path = os.path.join(lab_dir, f"{base_name}_gt.png")
        if not os.path.exists(lab_path):
            print(f"Missing label for {base_name}, skipping...")
            continue

        # Load 2D image and label
        img = np.array(Image.open(img_path)).astype(np.float32)
        lab = np.array(Image.open(lab_path)).astype(np.uint8)
        lab[lab == 255] = 1
        print(f"unique_label: {np.unique(lab)}")
        print(lab.shape)
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

# # source_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/pool_data/benign"
# # target_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/list"
# # create_yaml(source_path, train_ratio=0.8, seed=42, save_dir=target_path)
# # yaml_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/list/train.yaml"
# # source_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/pool_data/benign"
# # target_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/rescale_data/train"
# # copy_yaml_file(yaml_path, source_path, source_path, target_path)
# # organize_images_masks_rename(source_path, target_path)
# # source_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/CXR_png"
# # create_yaml(source_path, train_ratio=0.8, seed=42, save_dir="/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/list")
# # yaml_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/list/test.yaml"
# # source_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/CXR_png"
# # mask_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/masks"
# # target_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/rescale_data/test"
# #
# # copy_yaml_file(yaml_path, source_path, mask_path, target_path)
dataset_info = ('BUS', 'untrosound')
# source_path = "/research/cbim/medical/bg654/verse_dataset/Uwaterloo_skin_cancer/rescale_data"
# target_path = "/research/cbim/medical/bg654/verse_dataset/Uwaterloo_skin_cancer/Data/test"
# # copy_file(source_path , target_path, surname="_nm_dermquest")
# process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# dataset_info = ('chest_xray', 'mri')
source_path = "/common/users/bg654/verse_dataset/BUS"
target_path = "/common/users/bg654/verse_dataset/BUS/Data/"
# process_existing_2d_data(dataset_info, source_path, target_path, file="train")
process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# process_3d_data(dataset_info, source_path, target_path, file="train")
# process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'
