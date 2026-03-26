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

rename_masks_to_gt("/common/users/bg654/verse_dataset/BraTS/Data/test/annotations")

def convert_and_rename_gif_to_png(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith("_training.tif"):
            old_path = os.path.join(folder_path, file)
            new_name = file.replace("_training.tif", ".png")
            new_path = os.path.join(folder_path, new_name)

            # Load and convert
            with Image.open(old_path) as img:
                img = img.convert("RGB")  # or "L" if grayscale
                img.save(new_path, "PNG")

            print(f"Converted and saved: {file} → {new_name}")
            os.remove(old_path)  # Optional: remove original .gif
# 用法
# convert_and_rename_gif_to_png("/common/users/bg654/verse_dataset/DRIVE/rescale_data/test/image")

def process_existing_2d_data(source_path, file="test"):
    file_path = os.path.join(source_path, file)
    img_dir = os.path.join(file_path, "image")
    lab_dir = os.path.join(file_path, "annotations")

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
        # lab[lab == 255] = 1
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
        # imageio.imwrite(os.path.join(save_img_dir, f"{base_name}.png"), img)
        # imageio.imwrite(os.path.join(save_lab_dir, f"{base_name}_gt.png"), lab)

        print(f"✅ Processed {base_name}")
# process_existing_2d_data("/common/users/bg654/verse_dataset/BraTS/Data")
# # dataset_info = ('chest_xray', 'mri')
# source_path = "/common/users/bg654/verse_dataset/BUS"
# target_path = "/common/users/bg654/verse_dataset/BUS/Data/"
# # process_existing_2d_data(dataset_info, source_path, target_path, file="train")
# process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# process_3d_data(dataset_info, source_path, target_path, file="train")
# process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'
